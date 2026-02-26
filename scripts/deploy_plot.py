#!/usr/bin/env python3
"""
用已有的 processed episode + 训练好的 ckpt 跑一次离线推理，
对比预测 action 与 GT action，保存为 PNG。

Usage:
    python deploy_plot.py \
        --taskid 00 --exptid 00 \
        --ckpt policy_best.ckpt \
        --episode processed_episode_0.hdf5 \
        --chunk_size 50 \
        --state_dim 16 --action_dim 14 \
        --temporal_agg \
        --save result.png
"""

import argparse
import pickle
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.append(str(Path(__file__).resolve().parent.parent / 'act'))
from policy import ACTPolicy
from utils import parse_id

current_dir = Path(__file__).parent.resolve()
DATA_DIR   = (current_dir.parent / 'data/').resolve()
RECORD_DIR = (DATA_DIR / 'recordings/').resolve()
LOG_DIR    = (DATA_DIR / 'logs/').resolve()

CAMERA_NAMES = ['left', 'right', 'middle']


def get_norm_stats(stats_path):
    with open(stats_path, 'rb') as f:
        return pickle.load(f)


def load_policy(ckpt_path, args):
    policy_config = {
        'lr': 1e-5,
        'num_queries':      args['chunk_size'],
        'kl_weight':        10,
        'hidden_dim':       args['hidden_dim'],
        'dim_feedforward':  args['dim_feedforward'],
        'lr_backbone':      1e-5,
        'backbone':         'dino_v2',
        'enc_layers':       args['enc_layers'],
        'dec_layers':       args['dec_layers'],
        'nheads':           args['nheads'],
        'camera_names':     CAMERA_NAMES,
        'state_dim':        args['state_dim'],
        'action_dim':       args['action_dim'],
        'qpos_noise_std':   0,
    }
    # detr/main.py 的 build_ACT_model_and_optimizer 内部调用 parser.parse_args()，
    # 会读取 sys.argv 并要求 --policy_class / --seed / --num_epochs 等参数。
    # 临时替换 sys.argv 提供这些占位值（会被 args_override 覆盖，不影响实际配置）。
    orig_argv = sys.argv.copy()
    sys.argv = ['', '--policy_class', 'ACT', '--seed', '0', '--num_epochs', '1',
                '--taskid', args['taskid'], '--exptid', args['exptid']]
    policy = ACTPolicy(policy_config)
    sys.argv = orig_argv

    policy.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
    policy.cuda()
    policy.eval()
    return policy


def normalize_input(state, imgs_dict, norm_stats, state_dim):
    """
    state    : np.ndarray (state_dim,)  float32
    imgs_dict: {cam: np.ndarray (3,H,W) uint8}
    """
    # 旧模型 state_dim=16 只用 joint_pos；新模型 state_dim=30 用 joint_pos+ee_pose
    state = state[:state_dim].astype(np.float32)

    image_data = torch.from_numpy(
        np.stack([imgs_dict[c] for c in CAMERA_NAMES], axis=0)
    ).float() / 255.0                              # (3, 3, H, W)
    image_data = image_data.unsqueeze(0).cuda()    # (1, 3, 3, H, W)

    qpos_data = (torch.from_numpy(state)
                 - torch.from_numpy(norm_stats['qpos_mean'])
                 ) / torch.from_numpy(norm_stats['qpos_std'])
    qpos_data = qpos_data.float().unsqueeze(0).cuda()  # (1, state_dim)

    return qpos_data, image_data


def merge_act(all_time_actions, t, k=0.01):
    actions = all_time_actions[:t + 1, t]              # (t+1, action_dim)
    populated = np.any(actions != 0, axis=1)
    actions = actions[populated]
    exp_w = np.exp(-k * np.arange(len(actions)))
    exp_w = (exp_w / exp_w.sum()).reshape(-1, 1)
    return (actions * exp_w).sum(axis=0)


def run_inference(policy, norm_stats, state, imgs, args):
    T          = state.shape[0]
    chunk_size = args['chunk_size']
    action_dim = args['action_dim']
    state_dim  = args['state_dim']

    if args['temporal_agg']:
        all_time_actions = np.zeros([T, T + chunk_size, action_dim])

    pred_actions = np.zeros([T, action_dim])

    for t in tqdm(range(T), desc='inference'):
        qpos_data, image_data = normalize_input(
            state[t], {c: imgs[c][t] for c in CAMERA_NAMES}, norm_stats, state_dim)

        with torch.inference_mode():
            output = policy(qpos_data, image_data)  # (1, chunk_size, action_dim)
        output = output[0].cpu().numpy()             # (chunk_size, action_dim)

        if args['temporal_agg']:
            all_time_actions[[t], t:t + chunk_size] = output
            act_normed = merge_act(all_time_actions, t)
        else:
            act_normed = output[0]

        act = act_normed * norm_stats['action_std'] + norm_stats['action_mean']
        pred_actions[t] = act

    return pred_actions


def save_plot(pred_actions, gt_actions, action_dim, episode_name, args):
    """
    只绘制 arm0 / arm1 的 XYZ 位置，兼容旧格式（quat, action_dim=14/16）
    和新格式（6D, action_dim=20）。
      旧格式: arm0 pos=[0:3], arm1 pos=[7:10]
      新格式: arm0 pos=[0:3], arm1 pos=[9:12]
    """
    ts = np.arange(len(pred_actions))

    # 自动推断 arm1 起始索引
    arm1_start = 9 if action_dim >= 20 else 7

    rows = [
        ('arm0 pos', [0, 1, 2],                                ['x', 'y', 'z']),
        ('arm1 pos', [arm1_start, arm1_start+1, arm1_start+2], ['x', 'y', 'z']),
    ]
    n_rows = len(rows)

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3 * n_rows))
    fig.suptitle(
        f'{episode_name}  state_dim={args["state_dim"]}  '
        f'action_dim={action_dim}  chunk={args["chunk_size"]}  '
        f'temporal_agg={args["temporal_agg"]}',
        fontsize=10)

    for row_i, (title, idxs, labels) in enumerate(rows):
        for col_i, (idx, lbl) in enumerate(zip(idxs, labels)):
            ax = axes[row_i, col_i]
            ax.plot(ts, gt_actions[:, idx],   color='tab:blue',   lw=1.5, label='GT')
            ax.plot(ts, pred_actions[:, idx], color='tab:orange', lw=1.5,
                    linestyle='--', label='Pred')
            ax.set_title(f'{title}  {lbl}')
            ax.set_xlabel('step')
            ax.grid(True, alpha=0.3)
            if col_i == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args['save'], dpi=150)
    print(f'已保存 → {args["save"]}')


def save_plot_3d(pred_actions, gt_actions, action_dim, episode_name, save_path):
    """
    3D 轨迹图：arm0 / arm1 末端位置在三维空间的路径对比（GT vs 预测）。
    颜色由浅到深表示时间由早到晚。
    """
    # 根据 action_dim 确定位置索引
    # 旧格式(quat): arm0=[0:3], arm1=[7:10]
    # 新格式(6D):   arm0=[0:3], arm1=[9:12]
    arm1_start = 9 if action_dim >= 20 else 7

    arms = [
        ('arm0', slice(0, 3),           slice(0, 3)),
        ('arm1', slice(arm1_start, arm1_start + 3),
                 slice(arm1_start, arm1_start + 3)),
    ]

    T = len(pred_actions)

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f'{episode_name}  3D EE trajectory  (action_dim={action_dim})',
                 fontsize=11)

    for col, (arm_name, pred_sl, gt_sl) in enumerate(arms):
        ax = fig.add_subplot(1, 2, col + 1, projection='3d')

        gx, gy, gz = gt_actions[:, gt_sl].T
        px, py, pz = pred_actions[:, pred_sl].T

        # GT：实线 + 彩色散点
        ax.plot(gx, gy, gz, color='tab:blue', lw=1.0, alpha=0.5, label='GT')
        ax.scatter(gx, gy, gz, c=range(T), cmap='Blues', s=8, depthshade=False)

        # Pred：虚线 + 彩色散点
        ax.plot(px, py, pz, color='tab:orange', lw=1.0,
                linestyle='--', alpha=0.5, label='Pred')
        ax.scatter(px, py, pz, c=range(T), cmap='Oranges', s=8, depthshade=False)

        # 起点 / 终点标记
        ax.scatter(*[gx[0]], *[gy[0]], *[gz[0]], color='green', s=60,
                   marker='o', zorder=5, label='start(GT)')
        ax.scatter(*[gx[-1]], *[gy[-1]], *[gz[-1]], color='red', s=60,
                   marker='*', zorder=5, label='end(GT)')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        ax.set_title(arm_name)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f'已保存 3D 图 → {save_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--taskid',          type=str, required=True)
    parser.add_argument('--exptid',          type=str, required=True)
    parser.add_argument('--ckpt',            type=str, default='policy_best.ckpt')
    parser.add_argument('--episode',         type=str, default='processed_episode_0.hdf5')
    parser.add_argument('--chunk_size',      type=int, default=50)
    parser.add_argument('--state_dim',       type=int, default=34,
                        help='34=新模型(joint_pos 16 + ee_pose_6d 18)')
    parser.add_argument('--action_dim',      type=int, default=20,
                        help='20=新模型(ee_pose_6d 18 + gripper 2)')
    parser.add_argument('--hidden_dim',      type=int, default=512)
    parser.add_argument('--enc_layers',      type=int, default=4)
    parser.add_argument('--dec_layers',      type=int, default=7)
    parser.add_argument('--nheads',          type=int, default=8)
    parser.add_argument('--dim_feedforward', type=int, default=3200)
    parser.add_argument('--temporal_agg',    action='store_true')
    parser.add_argument('--save',            type=str, default='deploy_plot.png')
    args = vars(parser.parse_args())

    # 定位文件
    task_dir, task_name = parse_id(RECORD_DIR, args['taskid'])
    exp_dir, _          = parse_id((LOG_DIR / task_name).resolve(), args['exptid'])

    norm_stats = get_norm_stats(Path(exp_dir) / 'dataset_stats.pkl')
    print(f'qpos_mean shape : {norm_stats["qpos_mean"].shape}')
    print(f'action_mean shape: {norm_stats["action_mean"].shape}')

    policy = load_policy(Path(exp_dir) / args['ckpt'], args)
    print(f'policy loaded from {args["ckpt"]}')

    # 读 episode
    episode_path = Path(task_dir) / 'processed' / args['episode']
    with h5py.File(str(episode_path), 'r') as f:
        joint_pos  = np.array(f['observation/state/joint_pos'], dtype=np.float32)
        ee_pose    = np.array(f['observation/state/ee_pose'],   dtype=np.float32)
        gt_actions = np.array(f['action/ee_pose'],              dtype=np.float32)
        imgs = {c: np.array(f[f'observation/image/{c}']) for c in CAMERA_NAMES}

    # 拼接 state：旧模型只用 joint_pos，新模型用 joint_pos+ee_pose
    state = np.concatenate([joint_pos, ee_pose], axis=-1)  # (T, 30)，normalize_input 内部截断

    T = state.shape[0]
    print(f'episode: {episode_path.name}  T={T}  '
          f'gt_action_dim={gt_actions.shape[1]}')

    # 推理
    pred_actions = run_inference(policy, norm_stats, state, imgs, args)

    # 时间序列图
    save_plot(pred_actions, gt_actions, args['action_dim'], episode_path.name, args)

    # 3D 轨迹图（文件名加 _3d 后缀）
    p = Path(args['save'])
    save_path_3d = str(p.parent / (p.stem + '_3d' + p.suffix))
    save_plot_3d(pred_actions, gt_actions, args['action_dim'],
                 episode_path.name, save_path_3d)


if __name__ == '__main__':
    main()
