#!/usr/bin/env python3
"""
Offline deployment node: replays a processed episode through the trained ACT policy
and publishes predicted actions via native ROS2.

Usage:
    python deploy_ros2.py \
        --taskid 00 --exptid 00 \
        --ckpt policy_best.ckpt \
        --episode processed_episode_0.hdf5 \
        --chunk_size 100 --hz 10 \
        --temporal_agg

Topics published:
    /act/arm0/pred_pose   (geometry_msgs/PoseStamped)  predicted arm0 EE pose
    /act/arm1/pred_pose   (geometry_msgs/PoseStamped)  predicted arm1 EE pose
    /act/arm0/gt_pose     (geometry_msgs/PoseStamped)  ground truth arm0 EE pose
    /act/arm1/gt_pose     (geometry_msgs/PoseStamped)  ground truth arm1 EE pose
    /act/gripper          (std_msgs/Float32MultiArray)  [grp0, grp1] predicted
    /act/action_raw       (std_msgs/Float32MultiArray)  full 20-dim denormed action

TF frames broadcast:
    arm_0/base_link → arm_0/pred_ee   predicted arm0 EE
    arm_1/base_link → arm_1/pred_ee   predicted arm1 EE
    arm_0/base_link → arm_0/gt_ee     ground truth arm0 EE
    arm_1/base_link → arm_1/gt_ee     ground truth arm1 EE

Action layout (20-dim, from post_process.py):
    [0:3]   arm0 pos (x,y,z)
    [3:9]   arm0 rot6d
    [9:12]  arm1 pos (x,y,z)
    [12:18] arm1 rot6d
    [18]    gripper0
    [19]    gripper1
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import h5py
import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float32MultiArray

sys.path.append(str(Path(__file__).resolve().parent.parent / 'act'))
from policy import ACTPolicy
from utils import parse_id

current_dir = Path(__file__).parent.resolve()
DATA_DIR    = (current_dir.parent / 'data/').resolve()
RECORD_DIR  = (DATA_DIR / 'recordings/').resolve()
LOG_DIR     = (DATA_DIR / 'logs/').resolve()


# ── helpers ──────────────────────────────────────────────────────────────────

def rot6d_to_quat(r6d: np.ndarray) -> np.ndarray:
    """6D rotation representation → quaternion [x, y, z, w].

    r6d: (6,) = [r1(3), r2(3)] — first two columns of the rotation matrix.
    """
    r1 = r6d[0:3].astype(np.float64)
    r2 = r6d[3:6].astype(np.float64)
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 - np.dot(r1, r2) * r1
    r2 = r2 / np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)  # (3, 3), columns are basis vectors

    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)


def load_policy(ckpt_path, state_dim, action_dim, chunk_size,
                hidden_dim=512, enc_layers=4, dec_layers=7,
                nheads=8, dim_feedforward=3200):
    policy_config = {
        'lr': 1e-5,
        'num_queries': chunk_size,
        'kl_weight': 10,
        'hidden_dim': hidden_dim,
        'dim_feedforward': dim_feedforward,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': enc_layers,
        'dec_layers': dec_layers,
        'nheads': nheads,
        'camera_names': ['left', 'right', 'middle'],
        'state_dim': state_dim,
        'action_dim': action_dim,
        'qpos_noise_std': 0,
    }
    policy = ACTPolicy(policy_config)
    policy.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
    policy.cuda()
    policy.eval()
    return policy


def merge_act(all_time_actions, t, k=0.01):
    """Temporal aggregation: exponentially weighted average of past predictions."""
    actions = all_time_actions[:t + 1, t]
    populated = np.any(actions != 0, axis=1)
    actions = actions[populated]
    weights = np.exp(-k * np.arange(len(actions)))
    weights = (weights / weights.sum()).reshape(-1, 1)
    return (actions * weights).sum(axis=0)


def make_pose_msg(node, pos, quat, frame_id):
    """Return a ROS2 PoseStamped message.
    pos: [x,y,z], quat: [x,y,z,w]
    """
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.pose.position.x    = float(pos[0])
    msg.pose.position.y    = float(pos[1])
    msg.pose.position.z    = float(pos[2])
    msg.pose.orientation.x = float(quat[0])
    msg.pose.orientation.y = float(quat[1])
    msg.pose.orientation.z = float(quat[2])
    msg.pose.orientation.w = float(quat[3])
    return msg


# ── Deploy class ──────────────────────────────────────────────────────────────

class DeployNode(Node):
    def __init__(self, args):
        super().__init__('act_deploy')

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Publishers
        self.pub_pred_arm0 = self.create_publisher(PoseStamped,       '/act/arm0/pred_pose', 10)
        self.pub_pred_arm1 = self.create_publisher(PoseStamped,       '/act/arm1/pred_pose', 10)
        self.pub_gt_arm0   = self.create_publisher(PoseStamped,       '/act/arm0/gt_pose',   10)
        self.pub_gt_arm1   = self.create_publisher(PoseStamped,       '/act/arm1/gt_pose',   10)
        self.pub_gripper   = self.create_publisher(Float32MultiArray, '/act/gripper',        10)
        self.pub_action    = self.create_publisher(Float32MultiArray, '/act/action_raw',     10)

        # Locate experiment directory
        task_dir, task_name = parse_id(RECORD_DIR, args['taskid'])
        exp_dir, _          = parse_id((LOG_DIR / task_name).resolve(), args['exptid'])

        # Load normalisation stats
        stats_path = Path(exp_dir) / 'dataset_stats.pkl'
        with open(stats_path, 'rb') as f:
            self.norm_stats = pickle.load(f)
        self.get_logger().info(f'Loaded norm stats from {stats_path}')

        # Load policy
        ckpt_path = Path(exp_dir) / args['ckpt']
        self.policy = load_policy(
            ckpt_path,
            state_dim=args['state_dim'],
            action_dim=args['action_dim'],
            chunk_size=args['chunk_size'],
            hidden_dim=args['hidden_dim'],
            enc_layers=args['enc_layers'],
            dec_layers=args['dec_layers'],
            nheads=args['nheads'],
            dim_feedforward=args['dim_feedforward'],
        )
        self.get_logger().info(f'Loaded policy from {ckpt_path}')

        # Load episode data
        episode_path = Path(task_dir) / 'processed' / args['episode']
        with h5py.File(str(episode_path), 'r') as f:
            # qpos: ee_pose(18) + gripper(2) = 20-dim  (matches training utils.py)
            ee_pose   = f['observation/state/ee_pose'][:].astype(np.float32)   # (T, 18)
            joint_pos = f['observation/state/joint_pos'][:].astype(np.float32) # (T, N)
            gripper   = joint_pos[:, [12, 14]]                                  # (T, 2)
            self.states = np.concatenate([ee_pose, gripper], axis=-1)          # (T, 20)

            self.gt_actions = f['action/ee_pose'][:].astype(np.float32)        # (T, 20)
            self.imgs = {
                cam: f[f'observation/image/{cam}'][:]
                for cam in ['left', 'right', 'middle']
            }
        self.T = self.states.shape[0]
        self.get_logger().info(f'Episode: {episode_path.name}  ({self.T} steps)')

        self.chunk_size   = args['chunk_size']
        self.action_dim   = args['action_dim']
        self.temporal_agg = args['temporal_agg']
        self.frame_id     = args['frame_id']
        self.hz           = args['hz']

        if self.temporal_agg:
            self.all_time_actions = np.zeros(
                [self.T, self.T + self.chunk_size, self.action_dim])

    def step(self, t):
        # ── Build inputs ────────────────────────────────────────────────────
        # qpos: ee_pose(18) + gripper(2) = 20-dim
        state = self.states[t]
        qpos = (torch.from_numpy(state) - torch.from_numpy(self.norm_stats['qpos_mean'])) \
               / torch.from_numpy(self.norm_stats['qpos_std'])
        qpos = qpos.float().unsqueeze(0).cuda()

        imgs = np.stack([self.imgs[cam][t] for cam in ['left', 'right', 'middle']], axis=0)
        image = torch.from_numpy(imgs).float() / 255.0
        image = image.unsqueeze(0).cuda()

        # ── Inference ───────────────────────────────────────────────────────
        with torch.inference_mode():
            pred = self.policy(qpos, image)        # (1, chunk_size, action_dim)
        pred = pred[0].cpu().numpy()               # (chunk_size, action_dim)

        # ── Temporal aggregation ─────────────────────────────────────────────
        if self.temporal_agg:
            self.all_time_actions[[t], t:t + self.chunk_size] = pred
            act_normed = merge_act(self.all_time_actions, t)
        else:
            act_normed = pred[0]

        # ── Denormalise ─────────────────────────────────────────────────────
        act = act_normed * self.norm_stats['action_std'] + self.norm_stats['action_mean']
        # act layout (20-dim): arm0_pos(3)+arm0_rot6d(6) | arm1_pos(3)+arm1_rot6d(6) | grp0 | grp1

        # ── Publish predicted poses + TF ─────────────────────────────────────
        stamp = self.get_clock().now().to_msg()
        q0_pred = rot6d_to_quat(act[3:9])
        q1_pred = rot6d_to_quat(act[12:18])
        self.pub_pred_arm0.publish(make_pose_msg(self, act[0:3],  q0_pred, 'arm_0/base_link'))
        self.pub_pred_arm1.publish(make_pose_msg(self, act[9:12], q1_pred, 'arm_1/base_link'))

        def make_tf(parent, child, pos, quat):
            tf = TransformStamped()
            tf.header.stamp    = stamp
            tf.header.frame_id = parent
            tf.child_frame_id  = child
            tf.transform.translation.x = float(pos[0])
            tf.transform.translation.y = float(pos[1])
            tf.transform.translation.z = float(pos[2])
            tf.transform.rotation.x = float(quat[0])
            tf.transform.rotation.y = float(quat[1])
            tf.transform.rotation.z = float(quat[2])
            tf.transform.rotation.w = float(quat[3])
            return tf

        self.tf_broadcaster.sendTransform([
            make_tf('arm_0/base_link', 'arm_0/pred_ee', act[0:3],  q0_pred),
            make_tf('arm_1/base_link', 'arm_1/pred_ee', act[9:12], q1_pred),
        ])

        # ── Publish gripper ──────────────────────────────────────────────────
        grp_msg = Float32MultiArray()
        grp_msg.data = [float(act[18]), float(act[19])]
        self.pub_gripper.publish(grp_msg)

        # ── Publish ground-truth poses + TF ──────────────────────────────────
        gt = self.gt_actions[t]
        q0_gt = rot6d_to_quat(gt[3:9])
        q1_gt = rot6d_to_quat(gt[12:18])
        self.pub_gt_arm0.publish(make_pose_msg(self, gt[0:3],  q0_gt, 'arm_0/base_link'))
        self.pub_gt_arm1.publish(make_pose_msg(self, gt[9:12], q1_gt, 'arm_1/base_link'))

        self.tf_broadcaster.sendTransform([
            make_tf('arm_0/base_link', 'arm_0/gt_ee', gt[0:3],  q0_gt),
            make_tf('arm_1/base_link', 'arm_1/gt_ee', gt[9:12], q1_gt),
        ])

        # ── Publish full action ──────────────────────────────────────────────
        act_msg = Float32MultiArray()
        act_msg.data = act.tolist()
        self.pub_action.publish(act_msg)

        self.get_logger().info(
            f't={t:4d}/{self.T} | '
            f'arm0 pos=[{act[0]:.3f},{act[1]:.3f},{act[2]:.3f}] | '
            f'arm1 pos=[{act[9]:.3f},{act[10]:.3f},{act[11]:.3f}] | '
            f'gripper=[{act[18]:.3f},{act[19]:.3f}]')

    def run(self):
        period = 1.0 / self.hz
        for t in range(self.T):
            t0 = time.time()
            self.step(t)
            elapsed = time.time() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.get_logger().info('Episode replay finished.')


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument('--taskid',  type=str, required=True)
    parser.add_argument('--exptid',  type=str, required=True)
    # Optional
    parser.add_argument('--ckpt',            type=str,   default='policy_best.ckpt')
    parser.add_argument('--episode',         type=str,   default='processed_episode_0.hdf5')
    parser.add_argument('--chunk_size',      type=int,   default=100)
    parser.add_argument('--hz',              type=float, default=50)
    parser.add_argument('--state_dim',       type=int,   default=20)   # ee_pose_6d(18) + gripper(2)
    parser.add_argument('--action_dim',      type=int,   default=20)   # ee_pose_6d(18) + gripper(2)
    parser.add_argument('--hidden_dim',      type=int,   default=512)
    parser.add_argument('--enc_layers',      type=int,   default=4)
    parser.add_argument('--dec_layers',      type=int,   default=7)
    parser.add_argument('--nheads',          type=int,   default=8)
    parser.add_argument('--dim_feedforward', type=int,   default=3200)
    parser.add_argument('--frame_id',        type=str,   default='base_link')
    parser.add_argument('--temporal_agg',    action='store_true')
    args = vars(parser.parse_args())

    rclpy.init()
    node = DeployNode(args)
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
