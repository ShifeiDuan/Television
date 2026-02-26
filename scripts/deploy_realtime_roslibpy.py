#!/usr/bin/env python3
"""
Real-time ACT inference via roslibpy + multiprocessing.

推理在独立子进程中运行，与 roslibpy/Twisted 完全不共享 GIL。

执行策略：
  --temporal_agg  : 每步推理，temporal aggregation（适合精细控制）
  (默认)          : Chunk 执行 —— 推理一次执行整个 chunk，后台流水线
                    推理下一 chunk（标准 ACT 部署方式）

Usage:
    python deploy_realtime_roslibpy.py \
        --taskid 00 --exptid 03 \
        --hz 30 --chunk_size 50 \
        --ros-host localhost --ros-port 9090
"""

import argparse
import base64
import os
import pickle
import sys
import threading
import time
from multiprocessing import Event, Process
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np
import roslibpy

sys.path.append(str(Path(__file__).resolve().parent.parent / 'act'))
from utils import parse_id

current_dir = Path(__file__).parent.resolve()
DATA_DIR   = (current_dir.parent / 'data/').resolve()
RECORD_DIR = (DATA_DIR / 'recordings/').resolve()
LOG_DIR    = (DATA_DIR / 'logs/').resolve()
ACT_DIR    = str(Path(__file__).resolve().parent.parent / 'act')

IMG_H, IMG_W = 480, 640
N_CAMS   = 3
QPOS_DIM = 20
ACT_DIM  = 20

# Shared memory layout:
#   [0           : IMG_BYTES)       → images uint8  (N_CAMS, 3, H, W)
#   [IMG_BYTES   : +QPOS_BYTES)     → qpos   float32 (QPOS_DIM,)
#   [PRED_OFFSET : +PRED_BYTES)     → pred   float32 (chunk_size, ACT_DIM)  ← set at runtime
IMG_BYTES  = N_CAMS * 3 * IMG_H * IMG_W    # 2 764 800
QPOS_BYTES = QPOS_DIM * 4                   # 80
PRED_OFFSET = IMG_BYTES + QPOS_BYTES        # 2 764 880


# ── 旋転転換 ──────────────────────────────────────────────────────────────────

def quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = quat / np.linalg.norm(quat)
    r1 = np.array([1-2*(qy*qy+qz*qz), 2*(qx*qy+qz*qw), 2*(qx*qz-qy*qw)], dtype=np.float32)
    r2 = np.array([2*(qx*qy-qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz+qx*qw)], dtype=np.float32)
    return np.concatenate([r1, r2])


def rot6d_to_quat(r6d: np.ndarray) -> np.ndarray:
    r1 = r6d[0:3].astype(np.float64)
    r2 = r6d[3:6].astype(np.float64)
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 - np.dot(r1, r2) * r1
    r2 = r2 / np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    R  = np.stack([r1, r2, r3], axis=1)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1)
        w = 0.25 / s; x = (R[2,1]-R[1,2])*s; y = (R[0,2]-R[2,0])*s; z = (R[1,0]-R[0,1])*s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2])
        w = (R[2,1]-R[1,2])/s; x = 0.25*s; y = (R[0,1]+R[1,0])/s; z = (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = 2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2])
        w = (R[0,2]-R[2,0])/s; x = (R[0,1]+R[1,0])/s; y = 0.25*s; z = (R[1,2]+R[2,1])/s
    else:
        s = 2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1])
        w = (R[1,0]-R[0,1])/s; x = (R[0,2]+R[2,0])/s; y = (R[1,2]+R[2,1])/s; z = 0.25*s
    return np.array([x, y, z, w], dtype=np.float32)


def merge_act(all_time_actions, t, k=0.01):
    actions   = all_time_actions[:t+1, t]
    populated = np.any(actions != 0, axis=1)
    actions   = actions[populated]
    weights   = np.exp(-k * np.arange(len(actions)))
    weights   = (weights / weights.sum()).reshape(-1, 1)
    return (actions * weights).sum(axis=0)


# ── Inference subprocess ───────────────────────────────────────────────────────

def _inference_worker(shm_name, obs_event, act_event, stop_event,
                      norm_stats, args):
    """
    独立子进程：不与主进程共享 GIL。
    等待 obs_event → 读 shm → GPU 推理 → 写 pred 到 shm → set act_event。

    写入 shm 的内容 (PRED_OFFSET 起):
      - temporal_agg=True  : 单个 denorm action (ACT_DIM,)
      - temporal_agg=False : 完整 denorm chunk  (chunk_size, ACT_DIM)
    """
    os.environ['XFORMERS_DISABLED'] = '1'

    import sys, time
    sys.path.append(ACT_DIR)

    import torch
    import numpy as np
    from multiprocessing import shared_memory as _shm_mod
    from policy import ACTPolicy

    shm = _shm_mod.SharedMemory(name=shm_name)

    chunk_size   = args['chunk_size']
    temporal_agg = args['temporal_agg']

    policy_config = {
        'lr': 1e-5, 'num_queries': chunk_size, 'kl_weight': 10,
        'hidden_dim': args['hidden_dim'], 'dim_feedforward': args['dim_feedforward'],
        'lr_backbone': 1e-5, 'backbone': 'dino_v2',
        'enc_layers': args['enc_layers'], 'dec_layers': args['dec_layers'],
        'nheads': args['nheads'],
        'camera_names': ['left', 'right', 'middle'],
        'state_dim': QPOS_DIM, 'action_dim': ACT_DIM, 'qpos_noise_std': 0,
    }
    ckpt_path = Path(args['exp_dir']) / args['ckpt']
    policy = ACTPolicy(policy_config)
    policy.load_state_dict(torch.load(str(ckpt_path), map_location='cuda'))
    policy.cuda()
    policy.eval()
    print(f'[infer] Loaded policy from {ckpt_path}', flush=True)

    qpos_mean = torch.from_numpy(norm_stats['qpos_mean']).float().cuda()
    qpos_std  = torch.from_numpy(norm_stats['qpos_std']).float().cuda()
    act_mean  = norm_stats['action_mean']
    act_std   = norm_stats['action_std']

    step_t = 0
    if temporal_agg:
        all_time_actions = np.zeros([10000, 10000 + chunk_size, ACT_DIM])

    # Warm-up
    _d_img  = torch.zeros(1, N_CAMS, 3, IMG_H, IMG_W, device='cuda')
    _d_qpos = torch.zeros(1, QPOS_DIM, device='cuda')
    with torch.inference_mode():
        _ = policy(_d_qpos, _d_img)
    torch.cuda.synchronize()
    print('[infer] Warm-up done. Ready.', flush=True)

    while not stop_event.is_set():
        if not obs_event.wait(timeout=1.0):
            continue
        obs_event.clear()

        t0 = time.time()
        imgs = np.ndarray((N_CAMS, 3, IMG_H, IMG_W), dtype=np.uint8,
                          buffer=shm.buf, offset=0).copy()
        qpos = np.ndarray((QPOS_DIM,), dtype=np.float32,
                          buffer=shm.buf, offset=IMG_BYTES).copy()

        img_t  = torch.from_numpy(imgs).float().div_(255.0).unsqueeze(0).cuda()
        qpos_t = ((torch.from_numpy(qpos).cuda() - qpos_mean) / qpos_std).unsqueeze(0)

        t1 = time.time()
        with torch.inference_mode():
            pred = policy(qpos_t, img_t)   # (1, chunk_size, ACT_DIM)
        torch.cuda.synchronize()
        t2 = time.time()
        pred = pred[0].cpu().numpy()       # (chunk_size, ACT_DIM)

        print(f'  [infer] prep={( t1-t0)*1000:.1f}ms  gpu={( t2-t1)*1000:.1f}ms',
              flush=True)

        if temporal_agg:
            # 写单个聚合 action
            all_time_actions[[step_t], step_t:step_t+chunk_size] = pred
            act_normed = merge_act(all_time_actions, step_t)
            act = (act_normed * act_std + act_mean).astype(np.float32)
            np.ndarray((ACT_DIM,), dtype=np.float32,
                       buffer=shm.buf, offset=PRED_OFFSET)[:] = act
        else:
            # 写完整 chunk（denorm）
            chunk_denorm = (pred * act_std + act_mean).astype(np.float32)
            np.ndarray((chunk_size, ACT_DIM), dtype=np.float32,
                       buffer=shm.buf, offset=PRED_OFFSET)[:] = chunk_denorm

        step_t += 1
        act_event.set()

    shm.close()
    print('[infer] Worker exited.', flush=True)


# ── ROS bridge (main process) ──────────────────────────────────────────────────

class RealtimeDeployer:
    def __init__(self, ros_client, shm, obs_event, act_event, args):
        self.client      = ros_client
        self.shm         = shm
        self.obs_event   = obs_event
        self.act_event   = act_event
        self.lock        = threading.Lock()
        self.chunk_size  = args['chunk_size']
        self.temporal_agg = args['temporal_agg']

        self.latest_imgs = {'left': None, 'right': None, 'middle': None}
        self.latest_arm0 = None
        self.latest_arm1 = None
        self.latest_grp0 = None
        self.latest_grp1 = None
        self.last_sent_grp0 = None
        self.last_sent_grp1 = None
        self.grp_threshold = 0.01

        # Publishers
        self.pub_arm0  = roslibpy.Topic(ros_client, '/arm_0/target_frame',
                                        'geometry_msgs/PoseStamped')
        self.pub_arm1  = roslibpy.Topic(ros_client, '/arm_1/target_frame',
                                        'geometry_msgs/PoseStamped')
        self.pub_grip0 = roslibpy.Topic(ros_client,
                                        '/grp_0/joint_group_position_controller/commands',
                                        'std_msgs/Float64MultiArray')
        self.pub_grip1 = roslibpy.Topic(ros_client,
                                        '/grp_1/joint_group_position_controller/commands',
                                        'std_msgs/Float64MultiArray')
        self.pub_raw   = roslibpy.Topic(ros_client, '/act/action_raw',
                                        'std_msgs/Float32MultiArray')

        # Subscribers
        cam_topics = {
            'left':   '/arm_0/camera/color/image_raw/compressed',
            'right':  '/arm_1/camera/color/image_raw/compressed',
            'middle': '/camera/camera/color/image_raw/compressed',
        }
        for cam, topic_name in cam_topics.items():
            topic = roslibpy.Topic(ros_client, topic_name,
                                   'sensor_msgs/CompressedImage',
                                   throttle_rate=int(1000 / args.get('cam_hz', 30)),
                                   queue_length=1)
            topic.subscribe(lambda m, c=cam: self._img_cb(m, c))

        for attr, topic_name, msg_type, cb in [
            ('arm0', '/robot/arm0/ee_pose',    'geometry_msgs/PoseStamped', self._arm0_cb),
            ('arm1', '/robot/arm1/ee_pose',    'geometry_msgs/PoseStamped', self._arm1_cb),
            ('grp0', '/grp_0/joint_states',    'sensor_msgs/JointState',    self._grp0_cb),
            ('grp1', '/grp_1/joint_states',    'sensor_msgs/JointState',    self._grp1_cb),
        ]:
            t = roslibpy.Topic(ros_client, topic_name, msg_type, queue_length=1)
            t.subscribe(cb)

        print('Subscribed to all topics. Waiting for observations...')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _img_cb(self, msg, cam):
        try:
            data = base64.b64decode(msg['data'])
            arr  = np.frombuffer(data, dtype=np.uint8)
            img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img  = cv2.resize(img, (IMG_W, IMG_H))
            with self.lock:
                self.latest_imgs[cam] = img.transpose(2, 0, 1)
        except Exception as e:
            print(f'[img_cb/{cam}] {e}')

    def _arm0_cb(self, msg):
        p, q = msg['pose']['position'], msg['pose']['orientation']
        with self.lock:
            self.latest_arm0 = np.array(
                [p['x'], p['y'], p['z'], q['x'], q['y'], q['z'], q['w']], dtype=np.float32)

    def _arm1_cb(self, msg):
        p, q = msg['pose']['position'], msg['pose']['orientation']
        with self.lock:
            self.latest_arm1 = np.array(
                [p['x'], p['y'], p['z'], q['x'], q['y'], q['z'], q['w']], dtype=np.float32)

    def _grp0_cb(self, msg):
        with self.lock:
            self.latest_grp0 = float(msg['position'][0])

    def _grp1_cb(self, msg):
        with self.lock:
            self.latest_grp1 = float(msg['position'][0])

    # ── 观测 → shm ────────────────────────────────────────────────────────────

    def _collect_and_send_obs(self) -> bool:
        with self.lock:
            imgs = {k: v for k, v in self.latest_imgs.items()}
            arm0, arm1 = self.latest_arm0, self.latest_arm1
            grp0, grp1 = self.latest_grp0, self.latest_grp1

        fake_grp0 = self.last_sent_grp0 if self.last_sent_grp0 is not None else self.latest_grp0
        fake_grp1 = self.last_sent_grp1 if self.last_sent_grp1 is not None else self.latest_grp1

        missing = [k for k, v in imgs.items() if v is None]
        if arm0 is None: missing.append('arm0_pose')
        if arm1 is None: missing.append('arm1_pose')
        if grp0 is None: missing.append('grp0')
        if grp1 is None: missing.append('grp1')
        if missing:
            print(f'Waiting for: {missing}')
            return False

        qpos = np.concatenate([
            arm0[0:3], quat_to_rot6d(arm0[3:7]),
            arm1[0:3], quat_to_rot6d(arm1[3:7]),
            np.array([fake_grp0, fake_grp1], dtype=np.float32),
        ])

        np.ndarray((N_CAMS, 3, IMG_H, IMG_W), dtype=np.uint8,
                   buffer=self.shm.buf, offset=0)[0] = imgs['left']
        np.ndarray((N_CAMS, 3, IMG_H, IMG_W), dtype=np.uint8,
                   buffer=self.shm.buf, offset=0)[1] = imgs['right']
        np.ndarray((N_CAMS, 3, IMG_H, IMG_W), dtype=np.uint8,
                   buffer=self.shm.buf, offset=0)[2] = imgs['middle']
        np.ndarray((QPOS_DIM,), dtype=np.float32,
                   buffer=self.shm.buf, offset=IMG_BYTES)[:] = qpos

        self.obs_event.set()
        return True

    # ── action → ROS ──────────────────────────────────────────────────────────

    def _publish(self, act: np.ndarray, t: int):
        now = time.time()

        def pose_msg(pos, rot6d, frame):
            q = rot6d_to_quat(rot6d)
            return {
                'header': {'frame_id': frame,
                           'stamp': {'sec': int(now),
                                     'nanosec': int((now % 1) * 1e9)}},
                'pose': {
                    'position':    {'x': float(pos[0]), 'y': float(pos[1]),
                                    'z': float(pos[2])},
                    'orientation': {'x': float(q[0]),   'y': float(q[1]),
                                    'z': float(q[2]),   'w': float(q[3])},
                }
            }

        self.pub_arm0.publish(roslibpy.Message(
            pose_msg(act[0:3],  act[3:9],  'arm_0/base_link')))
        self.pub_arm1.publish(roslibpy.Message(
            pose_msg(act[9:12], act[12:18], 'arm_1/base_link')))
        if self.last_sent_grp0 is None or abs(act[18] - self.last_sent_grp0) > self.grp_threshold:
            self.pub_grip0.publish(roslibpy.Message({'data': [float(act[18])]}))
            self.last_sent_grp0 = act[18]
            
        if self.last_sent_grp1 is None or abs(act[19] - self.last_sent_grp1) > self.grp_threshold:
            self.pub_grip1.publish(roslibpy.Message({'data': [float(act[19])]}))
            self.last_sent_grp1 = act[19]
        self.pub_raw.publish(roslibpy.Message({'data': act.tolist()}))

        print(f't={t:4d} | arm0=[{act[0]:.3f},{act[1]:.3f},{act[2]:.3f}] '
              f'arm1=[{act[9]:.3f},{act[10]:.3f},{act[11]:.3f}] '
              f'grip=[{act[18]:.3f},{act[19]:.3f}]')

    # ── 主循环 ────────────────────────────────────────────────────────────────

    def run(self, hz):
        period = 1.0 / hz
        print(f'Running @ {hz} Hz  |  chunk_size={self.chunk_size}'
              f'  |  temporal_agg={self.temporal_agg}')

        if self.temporal_agg:
            self._run_temporal_agg(period)
        else:
            self._run_chunk_exec(period)

    def _run_temporal_agg(self, period):
        """每步推理 + temporal aggregation（适合精细/反应控制）"""
        t = 0
        while self.client.is_connected:
            t0 = time.time()
            if not self._collect_and_send_obs():
                time.sleep(0.1)
                continue
            if not self.act_event.wait(timeout=5.0):
                print('WARNING: inference timeout')
                continue
            self.act_event.clear()
            act = np.ndarray((ACT_DIM,), dtype=np.float32,
                             buffer=self.shm.buf, offset=PRED_OFFSET).copy()
            self._publish(act, t)
            t += 1
            sleep_t = period - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _run_chunk_exec(self, period):
            print('Waiting for initial observations...')
            while self.client.is_connected:
                if self._collect_and_send_obs():
                    break
                time.sleep(0.1)

            print('Waiting for first chunk (includes model warm-up)...')
            if not self.act_event.wait(timeout=30.0):
                print('ERROR: first inference timed out')
                return
            self.act_event.clear()
            
            # 拿到底一个 chunk，并触发下一次推理
            first_chunk = np.ndarray((self.chunk_size, ACT_DIM), dtype=np.float32,
                                    buffer=self.shm.buf, offset=PRED_OFFSET).copy()
            
            # 【核心 1】：轨迹队列，里面存放元组 (生成这个轨迹时的物理步数, 轨迹数据)
            trajectory_queue = [(0, first_chunk)]
            
            # 立刻让 GPU 去算下一个
            self._collect_and_send_obs()
            print('Got first chunk. Real ACT Temporal Ensembling Started!')

            t = 0
            k = 0.01  # ACT 原版的指数加权系数 (Exponential weighting)

            while self.client.is_connected:
                t_start = time.time()

                # 【核心 2】：非阻塞地获取新轨迹。
                # 只要 GPU 算完了（act_event 被 set），我们就把新轨迹加入队列，并立刻让 GPU 继续算
                if self.act_event.is_set():
                    new_chunk = np.ndarray((self.chunk_size, ACT_DIM), dtype=np.float32,
                                        buffer=self.shm.buf, offset=PRED_OFFSET).copy()
                    trajectory_queue.append((t, new_chunk))
                    self.act_event.clear()
                    self._collect_and_send_obs()  # 疯狂榨干 RTX 2080 的算力

                if not trajectory_queue:
                    time.sleep(0.001)
                    continue

                # 【核心 3】：收集当前时刻 t，所有轨迹给出的“建议动作”
                current_actions = []
                valid_trajectories = []
                
                for start_t, chunk in trajectory_queue:
                    idx = t - start_t
                    # 如果这个轨迹在当前时刻还有效（没有走完 50 步）
                    if 0 <= idx < self.chunk_size:
                        current_actions.append(chunk[idx])
                        valid_trajectories.append((start_t, chunk))
                
                # 清理掉那些已经超过 50 步、彻底作废的旧轨迹
                trajectory_queue = valid_trajectories

                # 【核心 4】：ACT 的灵魂 —— 加权平均
                if current_actions:
                    # 生成权重: 越早做出的预测权重越高（这能防止动作剧烈跳变）
                    weights = np.exp(-k * np.arange(len(current_actions)))
                    weights = weights / weights.sum()
                    
                    # 算出融合后的最终丝滑动作
                    fused_action = np.zeros(ACT_DIM)
                    for i, act in enumerate(current_actions):
                        fused_action += act * weights[i]

                    # 下发给底层 ROS (配合你之前的 0.0015 夹爪阈值，绝配！)
                    self._publish(fused_action, t)

                t += 1

                # 【核心 5】：严格把控物理时钟
                elapsed = time.time() - t_start
                sleep_t = period - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--taskid',          type=str,   required=True)
    parser.add_argument('--exptid',          type=str,   required=True)
    parser.add_argument('--ckpt',            type=str,   default='policy_best.ckpt')
    parser.add_argument('--hz',              type=float, default=30.0)
    parser.add_argument('--chunk_size',      type=int,   default=50)
    parser.add_argument('--hidden_dim',      type=int,   default=512)
    parser.add_argument('--enc_layers',      type=int,   default=4)
    parser.add_argument('--dec_layers',      type=int,   default=7)
    parser.add_argument('--nheads',          type=int,   default=8)
    parser.add_argument('--dim_feedforward', type=int,   default=3200)
    parser.add_argument('--temporal_agg',    action='store_true')
    parser.add_argument('--cam_hz',          type=int,   default=15)
    parser.add_argument('--ros-host',        type=str,   default='localhost')
    parser.add_argument('--ros-port',        type=int,   default=9090)
    args = vars(parser.parse_args())

    # 找 checkpoint 目录
    _, task_name = parse_id(RECORD_DIR, args['taskid'])
    exp_dir, _   = parse_id((LOG_DIR / task_name).resolve(), args['exptid'])
    args['exp_dir'] = str(exp_dir)

    with open(Path(exp_dir) / 'dataset_stats.pkl', 'rb') as f:
        norm_stats = pickle.load(f)
    print('Loaded norm stats.')

    # 动态计算 shm 大小（chunk_size 是运行时参数）
    pred_bytes = args['chunk_size'] * ACT_DIM * 4
    shm_size   = IMG_BYTES + QPOS_BYTES + pred_bytes

    shm        = shared_memory.SharedMemory(create=True, size=shm_size)
    obs_event  = Event()
    act_event  = Event()
    stop_event = Event()

    # 启动推理子进程（fork 前主进程不触碰 CUDA）
    inf_proc = Process(
        target=_inference_worker,
        args=(shm.name, obs_event, act_event, stop_event, norm_stats, args),
        daemon=True,
    )
    inf_proc.start()

    # 连接 ROS
    client = roslibpy.Ros(host=args['ros_host'], port=args['ros_port'])
    client.run()
    timeout, t0 = 10.0, time.time()
    while not client.is_connected and time.time() - t0 < timeout:
        time.sleep(0.1)
    if not client.is_connected:
        print(f'ERROR: Cannot connect to rosbridge at {args["ros_host"]}:{args["ros_port"]}')
        stop_event.set(); inf_proc.join(timeout=3.0)
        shm.close(); shm.unlink(); client.terminate()
        return
    print(f'Connected to rosbridge at {args["ros_host"]}:{args["ros_port"]}')

    deployer = RealtimeDeployer(client, shm, obs_event, act_event, args)
    try:
        deployer.run(args['hz'])
    except KeyboardInterrupt:
        print('Interrupted.')
    finally:
        stop_event.set()
        inf_proc.join(timeout=5.0)
        shm.close()
        shm.unlink()
        client.terminate()


if __name__ == '__main__':
    main()
