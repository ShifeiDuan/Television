#!/usr/bin/env python3
"""
Post-process recorded episodes from DualArmJointRecorder.

Input per episode (same base name):
  <base>_left.mp4
  <base>_right.mp4
  <base>_middle.mp4
  <base>.hdf5

Output:
  <path>/processed/processed_<base>.hdf5
    observation/image/left    (T, 3, H, W)  uint8
    observation/image/right   (T, 3, H, W)  uint8
    observation/image/middle  (T, 3, H, W)  uint8
    observation/state/ee_pose    (T, 18)  float32  [arm0: pos(3)+rot6d(6), arm1: pos(3)+rot6d(6)]
    observation/state/joint_pos  (T, N)   float32
    observation/timestamp        (T,)     float64  [seconds, ROS clock]
    action/ee_pose               (T, 20)  float32  [arm0: pos(3)+rot6d(6), arm1: pos(3)+rot6d(6), grp0, grp1]
    attrs: episode, num_steps, sim, + original metadata

NOTE: Video frames and HDF5 steps are recorded in the same timer loop,
      so frame N <-> step N by index — no timestamp matching is needed.

Usage:
    # Process all episodes in a directory
    python post_process.py --path ./trajectories

    # Process a single episode
    python post_process.py --path ./trajectories --episode episode_20250101_120000

    # Parallel processing
    python post_process.py --path ./trajectories --parallel
"""

import argparse
import concurrent.futures
import os
import time
from pathlib import Path

import cv2
import h5py
import numpy as np


# ── 旋转表示转换 ──────────────────────────────────────────────────────────────

def quat_to_6d(q: np.ndarray) -> np.ndarray:
    """
    四元数 [qx, qy, qz, qw] → 6D 旋转表示（旋转矩阵的前两列拼接）。
    6D 旋转连续、无符号歧义，适合神经网络学习 (Zhou et al. 2019)。
    """
    q = q / np.linalg.norm(q)          # 防止数值漂移导致非单位四元数
    qx, qy, qz, qw = q
    # 旋转矩阵第 0 列 (r1) 和第 1 列 (r2)
    r1 = np.array([1 - 2*(qy*qy + qz*qz),
                       2*(qx*qy + qz*qw),
                       2*(qx*qz - qy*qw)], dtype=np.float32)
    r2 = np.array([    2*(qx*qy - qz*qw),
                   1 - 2*(qx*qx + qz*qz),
                       2*(qy*qz + qx*qw)], dtype=np.float32)
    return np.concatenate([r1, r2])     # (6,)


def ee_pose_to_6d(ee_pose: np.ndarray) -> np.ndarray:
    """
    将整段 ee_pose 序列从四元数格式转换为 6D 格式。

    Input : (T, 14)  [arm0_pos(3), arm0_quat(4), arm1_pos(3), arm1_quat(4)]
    Output: (T, 18)  [arm0_pos(3), arm0_6d(6),   arm1_pos(3), arm1_6d(6)]
    """
    T = ee_pose.shape[0]
    out = np.zeros((T, 18), dtype=np.float32)
    for t in range(T):
        out[t, 0:3]   = ee_pose[t, 0:3]               # arm0 pos
        out[t, 3:9]   = quat_to_6d(ee_pose[t, 3:7])   # arm0 rot6d
        out[t, 9:12]  = ee_pose[t, 7:10]              # arm1 pos
        out[t, 12:18] = quat_to_6d(ee_pose[t, 10:14]) # arm1 rot6d
    return out


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_video_frames(mp4_path: str) -> np.ndarray:
    """Load all frames from an MP4 file.

    Returns
    -------
    np.ndarray  shape (T, 3, H, W), dtype uint8, channel order RGB
    """
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {mp4_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # HWC, RGB
        frames.append(rgb.transpose(2, 0, 1))              # -> CHW
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from: {mp4_path}")
    return np.array(frames, dtype=np.uint8)                # (T, 3, H, W)


def load_hdf5(hdf5_path: str):
    """Load joint/EE data from the recorder's HDF5.

    Returns
    -------
    timestamps      : (T,)    float64  seconds
    ee_pose         : (T, 14) float32
    joint_pos       : (T, N)  float32
    action_ee       : (T, 14) float32
    action_gripper  : (T, 2)  float32
    meta            : dict    original metadata attributes
    """
    with h5py.File(hdf5_path, 'r') as f:
        timestamps     = np.array(f['obs/timestamp'][:],       dtype=np.float64)
        ee_pose        = np.array(f['obs/ee_pose'][:],         dtype=np.float32)
        joint_pos      = np.array(f['obs/joint_pos'][:],       dtype=np.float32)
        action_ee      = np.array(f['action/ee_pose'][:],      dtype=np.float32)
        action_gripper = np.array(f['action/gripper_cmd'][:],  dtype=np.float32)

        meta = {}
        if 'metadata' in f:
            for k, v in f['metadata'].attrs.items():
                meta[k] = v

    return timestamps, ee_pose, joint_pos, action_ee, action_gripper, meta


# ── Resampling ────────────────────────────────────────────────────────────────

def resample_to_uniform_hz(timestamps, ee_pose, joint_pos, action_ee, action_gripper,
                            imgs_left, imgs_right, imgs_middle,
                            target_hz: float = 50.0):
    """把所有数据重采样到均匀的 target_hz 网格。

    标量数据（ee_pose / joint_pos / action_ee / action_gripper）线性插值；
    图像用最近邻（不做帧间混合）；四元数部分插值后归一化。
    """
    t0, t1 = timestamps[0], timestamps[-1]
    n_new = max(2, round((t1 - t0) * target_hz) + 1)
    t_new = np.linspace(t0, t1, n_new)

    # ── 标量线性插值 ──────────────────────────────────────────────────────
    def interp(src):
        out = np.empty((n_new,) + src.shape[1:], dtype=np.float32)
        for d in np.ndindex(src.shape[1:]):
            idx = (slice(None),) + d
            out[idx] = np.interp(t_new, timestamps,
                                 src[idx].astype(np.float64))
        return out

    ee_r   = interp(ee_pose)
    jpos_r = interp(joint_pos)
    act_r  = interp(action_ee)
    grp_r  = interp(action_gripper)

    # 四元数归一化（arm0: [3:7], arm1: [10:14]）
    for s in (slice(3, 7), slice(10, 14)):
        for arr in (ee_r, act_r):
            norms = np.linalg.norm(arr[:, s], axis=1, keepdims=True)
            arr[:, s] /= np.where(norms > 0, norms, 1.0)

    # ── 图像最近邻 ────────────────────────────────────────────────────────
    idx = np.searchsorted(timestamps, t_new)
    idx = np.clip(idx, 0, len(timestamps) - 1)
    idx_l = np.clip(idx - 1, 0, len(timestamps) - 1)
    use_left = (np.abs(timestamps[idx_l] - t_new) <
                np.abs(timestamps[idx]   - t_new))
    idx[use_left] = idx_l[use_left]

    return (t_new, ee_r, jpos_r, act_r, grp_r,
            imgs_left[idx], imgs_right[idx], imgs_middle[idx])


# ── Episode discovery ─────────────────────────────────────────────────────────

def find_all_episodes(path: str):
    """Return sorted list of complete episode base names in *path*.

    An episode is considered complete only when all four files exist:
      <base>_joints.hdf5, <base>_left.mp4, <base>_right.mp4, <base>_middle.mp4
    """
    episodes = []
    for fname in os.listdir(path):
        if not fname.endswith('.hdf5'):
            continue
        base = fname[: -len('.hdf5')]
        required = [
            os.path.join(path, base + '_left.mp4'),
            os.path.join(path, base + '_right.mp4'),
            os.path.join(path, base + '_middle.mp4'),
        ]
        missing = [r for r in required if not os.path.exists(r)]
        if missing:
            print(f"[SKIP] Incomplete episode '{base}' — missing: {missing}")
        else:
            episodes.append(base)
    return sorted(episodes)


# ── Core processing ───────────────────────────────────────────────────────────

def process_episode(path: str, ep_base: str) -> str:
    """Process a single episode and save the merged HDF5.

    Parameters
    ----------
    path    : directory that contains the raw episode files
    ep_base : episode base name, e.g. 'episode_20250101_120000'

    Returns
    -------
    str : path to the saved processed HDF5 file
    """
    hdf5_path  = os.path.join(path, ep_base + '.hdf5')
    mp4_left   = os.path.join(path, ep_base + '_left.mp4')
    mp4_right  = os.path.join(path, ep_base + '_right.mp4')
    mp4_middle = os.path.join(path, ep_base + '_middle.mp4')

    print(f"\n{'='*60}")
    print(f"Processing: {ep_base}")

    # ── Load HDF5 ──────────────────────────────────────────────────────────
    timestamps, ee_pose, joint_pos, action_ee, action_gripper, meta = load_hdf5(hdf5_path)
    T_hdf5 = len(timestamps)
    print(f"  HDF5 steps : {T_hdf5}")
    delta = np.diff(timestamps)
    print(f"  Step interval  mean={np.mean(delta)*1000:.1f}ms  std={np.std(delta)*1000:.1f}ms")

    # ── Load videos ────────────────────────────────────────────────────────
    print("  Loading videos ...")
    imgs_left   = load_video_frames(mp4_left)
    imgs_right  = load_video_frames(mp4_right)
    imgs_middle = load_video_frames(mp4_middle)

    T_left, T_right, T_mid = len(imgs_left), len(imgs_right), len(imgs_middle)
    print(f"  Video frames : left={T_left}  right={T_right}  middle={T_mid}")

    # ── Trim to consistent length ──────────────────────────────────────────
    # Videos and HDF5 are written in the same loop → frame N == step N.
    # Lengths may differ by ±1 frame due to shutdown timing; take the minimum.
    T = min(T_hdf5, T_left, T_right, T_mid)
    if T < T_hdf5:
        print(f"  [INFO] Trimmed to T={T} (smallest of HDF5 and three videos)")

    imgs_left   = imgs_left[:T]
    imgs_right  = imgs_right[:T]
    imgs_middle = imgs_middle[:T]
    timestamps  = timestamps[:T]
    ee_pose     = ee_pose[:T]
    joint_pos   = joint_pos[:T]
    action_ee      = action_ee[:T]
    action_gripper = action_gripper[:T]

    # ── 重采样到均匀时间网格 ────────────────────────────────────────────────
    target_hz = float(meta.get('recording_frequency', 50.0))
    (timestamps, ee_pose, joint_pos, action_ee, action_gripper,
     imgs_left, imgs_right, imgs_middle) = resample_to_uniform_hz(
        timestamps, ee_pose, joint_pos, action_ee, action_gripper,
        imgs_left, imgs_right, imgs_middle,
        target_hz=target_hz,
    )
    T = len(timestamps)
    delta2 = np.diff(timestamps)
    print(f"  After resample : {T} steps @ {target_hz:.0f} Hz  "
          f"(duration={timestamps[-1]-timestamps[0]:.2f}s  "
          f"interval mean={np.mean(delta2)*1000:.1f}ms  std={np.std(delta2)*1000:.2f}ms)")

    # ── Save ───────────────────────────────────────────────────────────────
    out_dir = os.path.join(path, 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'processed_{ep_base}.hdf5')

    print(f"  Saving -> {out_path}")
    t0 = time.time()

    with h5py.File(out_path, 'w') as f:
        # Images  (T, 3, H, W)  uint8  RGB
        img_grp = f.create_group('observation/image')
        img_grp.create_dataset('left',   data=imgs_left,   compression='gzip', compression_opts=4)
        img_grp.create_dataset('right',  data=imgs_right,  compression='gzip', compression_opts=4)
        img_grp.create_dataset('middle', data=imgs_middle, compression='gzip', compression_opts=4)

        # State  (ee_pose 转为 6D 旋转表示)
        ee_pose_6d = ee_pose_to_6d(ee_pose)          # (T, 18)
        state_grp = f.create_group('observation/state')
        state_grp.create_dataset('ee_pose',   data=ee_pose_6d, compression='gzip')
        state_grp.create_dataset('joint_pos', data=joint_pos,  compression='gzip')

        # Timestamps (actual ROS wall-clock, seconds)
        f.create_dataset('observation/timestamp', data=timestamps, compression='gzip')

        # Action: ee_pose_6d(18) + gripper_cmd(2) = 20 dims
        action_ee_6d = ee_pose_to_6d(action_ee)                                    # (T, 18)
        action_ee_with_gripper = np.concatenate([action_ee_6d, action_gripper], axis=-1)  # (T, 20)
        act_grp = f.create_group('action')
        act_grp.create_dataset('ee_pose', data=action_ee_with_gripper, compression='gzip')

        # Top-level attributes
        f.attrs['episode']   = ep_base
        f.attrs['num_steps'] = T
        f.attrs['sim']       = False
        # Copy original recorder metadata
        for k, v in meta.items():
            try:
                f.attrs[k] = v
            except Exception:
                pass  # skip un-serialisable values

    elapsed = time.time() - t0
    print(f"  Saved {T} steps in {elapsed:.1f}s")
    print(f"  Image shape : {imgs_left.shape}  ({imgs_left.dtype})")
    print(f"  EE pose     : {ee_pose.shape}")
    print(f"  Joint pos   : {joint_pos.shape}")
    return out_path


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(processed_dir: Path) -> None:
    eps = sorted(processed_dir.glob('processed_*.hdf5'))
    if not eps:
        print("\nNo processed episodes found.")
        return

    print(f"\n{'='*60}")
    print(f"Processed episodes in: {processed_dir}")
    print(f"{'='*60}")
    lens = []
    for ep_path in eps:
        with h5py.File(str(ep_path), 'r') as f:
            T   = int(f.attrs.get('num_steps', f['observation/timestamp'].shape[0]))
            img = f['observation/image/left']
            shape_str = f"{'x'.join(str(s) for s in img.shape[1:])}"
        print(f"  {ep_path.name:<55s}  {T:4d} steps  img {shape_str}")
        lens.append(T)
    lens = np.array(lens)
    print(f"\nTotal: {len(lens)} episodes  |  steps min={lens.min()}  max={lens.max()}  mean={lens.mean():.0f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Post-process dual-arm recorder episodes into merged HDF5 files.')
    parser.add_argument('--path',     type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'recordings', '01-pt'),
                        help='Directory containing raw episode files')
    parser.add_argument('--episode',  type=str, default=None,
                        help='Process a single episode base name (optional)')
    parser.add_argument('--parallel', action='store_true',
                        help='Use multiprocess parallel processing')
    args = parser.parse_args()

    path = args.path

    if args.episode:
        all_eps = [args.episode]
    else:
        all_eps = find_all_episodes(path)

    if not all_eps:
        print("No complete episodes found. Exiting.")
        return

    print(f"Found {len(all_eps)} episode(s):")
    for ep in all_eps:
        print(f"  {ep}")

    if args.parallel and len(all_eps) > 1:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_map = {executor.submit(process_episode, path, ep): ep
                          for ep in all_eps}
            for future in concurrent.futures.as_completed(future_map):
                ep = future_map[future]
                try:
                    out = future.result()
                    print(f"[OK]   {ep} -> {out}")
                except Exception as exc:
                    print(f"[FAIL] {ep}: {exc}")
    else:
        for ep in all_eps:
            try:
                out = process_episode(path, ep)
                print(f"[OK]   -> {out}")
            except Exception as exc:
                print(f"[FAIL] {ep}: {exc}")

    print_summary(Path(path) / 'processed')


if __name__ == '__main__':
    main()

# import h5py
# import numpy as np
# import pyzed.sl as sl
# import time
# import cv2
# import matplotlib.pyplot as plt 
# import tqdm
# import torch
# from torch.utils.data import Dataset
# import os 
# import multiprocessing
# from numpy.lib.stride_tricks import as_strided

# from pytransform3d import rotations
# import concurrent.futures
# from pathlib import Path
# import argparse

# def load_svo(path, crop_size_h=240, crop_size_w=320):
#     input_file = path + ".svo"
#     # import ipdb; ipdb.set_trace()
#     print(input_file)
#     crop_size_h = crop_size_h
#     crop_size_w = crop_size_w
#     init_parameters = sl.InitParameters()
#     init_parameters.set_from_svo_file(input_file)

#     zed = sl.Camera()
#     err = zed.open(init_parameters)
#     left_image = sl.Mat()
#     right_image = sl.Mat()

#     nb_frames = zed.get_svo_number_of_frames()
#     print("Total image frames: ", nb_frames)

#     cropped_img_shape = (720-crop_size_h, 1280-2*crop_size_w)
#     left_imgs = np.zeros((nb_frames, 3, cropped_img_shape[0], cropped_img_shape[1]), dtype=np.uint8)
#     right_imgs = np.zeros((nb_frames, 3, cropped_img_shape[0], cropped_img_shape[1]), dtype=np.uint8)
#     timestamps = np.zeros((nb_frames, ), dtype=np.int64)
#     cnt = 0
#     while True:
#         if zed.grab() == sl.ERROR_CODE.SUCCESS:
#             zed.retrieve_image(left_image, sl.VIEW.LEFT)
#             zed.retrieve_image(right_image, sl.VIEW.RIGHT)

#             timestamps[cnt] = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
#             # import ipdb; ipdb.set_trace()
#             left_imgs[cnt] = cv2.cvtColor(left_image.get_data()[crop_size_h:, crop_size_w:-crop_size_w], cv2.COLOR_BGRA2RGB).transpose(2, 0, 1)
#             right_imgs[cnt] = cv2.cvtColor(right_image.get_data()[crop_size_h:, crop_size_w:-crop_size_w], cv2.COLOR_BGRA2RGB).transpose(2, 0, 1)
#             cnt += 1
#             if cnt % 100 == 0:
#                 print(f"{cnt/nb_frames*100:.2f}%")
#                 # plt.imsave(f"left_img_{cnt}.png", left_imgs[cnt-1].transpose(1, 2, 0))
#         elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
#             break
#     # print delta mean and std for img_timstamps
#     delta = np.diff(timestamps)[:-1]
#     print("img timestamps delta mean: ", np.mean(delta))
#     print("img timestamps delta std: ", np.std(delta))
#     return left_imgs[10:-10], right_imgs[10:-10], timestamps[10:-10]

# def load_hdf5(path, offset=10):  # offset 10ms
#     input_file = path + ".hdf5"
#     file = h5py.File(input_file, 'r')
#     print(f"Total hdf5_frames: {file['/obs/timestamp'].shape[0]}")
#     # print(file["/obs/timestamp"].shape)
#     # print(file["/obs/qpos"].shape)
#     # print(file["/obs/qvel"].shape)
#     # print(file["/action/joint_pos"].shape)
#     # print("keys: ", list(file.keys()))
#     timestamps = np.array(file["/obs/timestamp"][:] * 1000, dtype=np.int64) - offset
#     states = np.array(file["/obs/qpos"][:])
#     actions = np.array(file["/action/joint_pos"][:])
#     cmds = np.array(file["/action/cmd"][:])

#     return timestamps, states, actions, cmds

# def match_timestamps(candidate, ref):
#     closest_indices = []
#     # candidate = np.sort(candidate)
#     for t in ref:
#         idx = np.searchsorted(candidate, t, side="left")
#         if idx > 0 and (idx == len(candidate) or np.fabs(t - candidate[idx-1]) < np.fabs(t - candidate[idx])):
#             closest_indices.append(idx-1)
#         else:
#             closest_indices.append(idx)
#     # print("closest_indices: ", len(closest_indices))
#     return np.array(closest_indices)

# def find_all_episodes(path):
#     episodes = [os.path.join(path, f) for f in os.listdir(path) if f.startswith('episode') and f.endswith('.svo')]
#     episodes = [os.path.basename(ep).split(".")[0] for ep in episodes]
#     return episodes

# def create_chunks(data, chunk_size):
#     N, F = data.shape
#     if chunk_size > N:
#         raise ValueError("chunk_size cannot be greater than N.")
    
#     stride0, stride1 = data.strides
#     new_shape = (N - chunk_size + 1, chunk_size, F)
#     new_strides = (stride0, stride0, stride1)
    
#     return as_strided(data, shape=new_shape, strides=new_strides)

# def process_episode(file_name, ep):
#     left_imgs, right_imgs, img_timestamps = load_svo(file_name)
#     hdf5_timestamps, states, actions, cmds = load_hdf5(file_name)
#     closest_indices = match_timestamps(candidate=hdf5_timestamps, ref=img_timestamps)

#     timesteps = len(closest_indices)
#     qpos_actions = actions[closest_indices]
#     cmds = cmds[closest_indices]
    
#     # save_video(left_imgs, file_name + ".mp4")
#     path = os.path.dirname(file_name)
#     all_data_path = os.path.join(path, "processed")
#     os.makedirs(all_data_path, exist_ok=True)

#     with h5py.File(all_data_path + f"/processed_{ep}.hdf5", 'w') as hf:
#         start = time.time()
#         hf.create_dataset('observation.image.left', data=left_imgs)
#         hf.create_dataset('observation.image.right', data=right_imgs)
#         hf.create_dataset('cmds', data=cmds.astype(np.float32))
#         hf.create_dataset('observation.state', data=states[closest_indices].astype(np.float32))
#         hf.create_dataset('qpos_action', data=qpos_actions.astype(np.float32))
#         hf.attrs['sim'] = False
#         hf.attrs['init_action'] = cmds[0].astype(np.float32)
        
#         print("Time to save dataset: ", time.time() - start)

# def process_all_episodes(all_eps, path):
#     results = []
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         future_to_ep = {executor.submit(process_episode, os.path.join(path, ep), ep): ep for ep in all_eps}
#         for future in concurrent.futures.as_completed(future_to_ep):
#             ep = future_to_ep[future]
#             try:
#                 result = future.result()
#                 results.append(result)
#             except Exception as e:
#                 print(f"Episode {ep} generated an exception: {e}")
#     return results

# def save_video(left_imgs, path):
#     _, height, width= left_imgs[0].shape
#     print(f"width: {width}, height: {height}")
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video_writer = cv2.VideoWriter(path, fourcc, 60, (width, height))

#     for img in left_imgs:
#         # print(img.shape)
#         img_bgr = cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
#         video_writer.write(img_bgr)

#     video_writer.release()

# def find_all_processed_episodes(path):
#     episodes = [f for f in os.listdir(path)]
#     return episodes


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--save_video', action='store_true', default=False)
#     args = parser.parse_args()

#     root = "../data/recordings"
#     folder_name = "00-can-sorting"


#     path = os.path.join(root, folder_name)

#     all_eps = find_all_episodes(path)

#     if args.save_video:
#         file_name = path + "/" + all_eps[0]
#         print("saving video for file: ", file_name)
#         left_imgs, right_imgs, img_timestamps = load_svo(file_name)
#         os.makedirs(os.path.join(path, "videos"), exist_ok=True)
#         save_video(left_imgs, os.path.join(path, "videos", "sample.mp4"))
#     else:
#         for ep in all_eps:
#             file_name = path + "/" + ep
#             process_episode(file_name, ep)
#             print('processed file', file_name)

#     # print len
#     folder_path = Path(root) / folder_name / "processed"

#     episodes = find_all_processed_episodes(folder_path)
#     num_episodes = len(episodes)
#     lens = []

#     for episode in episodes:
#         episode_path = folder_path / episode
        
#         data = h5py.File(str(episode_path), 'r')
#         lens.append(data['qpos_action'].shape[0])
#         data.close()

#     lens = np.array(lens)
#     episodes = np.array(episodes)
#     print(lens[np.argsort(lens)])
#     print(episodes[np.argsort(lens)])
#     # results = process_all_episodes(all_eps, path)
#     # print(len(results))
