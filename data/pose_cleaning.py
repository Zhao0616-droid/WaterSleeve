"""
姿态估计输出 -> EDGE 兼容动作 pkl 的清洗管线（模块一 数据底座）。

流程：加载(多格式) -> 坐标系转换(y-up) -> 重采样到 60fps -> 平滑(旋转矩阵空间高斯 / 位移低通)
     -> 接触检测(3D 关键点, 迟滞+地面高度) -> 脚步滑动修正 -> 输出 AIST++ 格式 pkl -> (可选) 5s 切窗

输出与 data/slice.py 约定完全一致：完整序列 {"pos": F×3(米), "q": F×72(轴角), "scale": [s]}，y-up 60fps；
切窗 {"pos": 300×3, "q": 300×72}。下游 dataset/dance_dataset.py 的 process_dataset
（y-up→z-up 旋转、SMPL FK、接触标签、6D 化、归一化）无需任何修改即可使用。

纯 numpy 实现（无 pytorch3d 依赖），平滑的 Butterworth 低通在无 scipy 时自动降级为高斯平滑。
接触标签语义与 EDGE 一致：4 通道 = [左脚踝, 右脚踝, 左脚, 右脚]（对应 SMPL 关节 7,8,10,11）。

接入训练示例：
    python data/pose_cleaning.py --input_dir poses_raw/ --output_dir data/custom_aistpp \
        --joints body25 --slice --save_contacts
    # 然后把 motions_sliced / wavs_sliced 与 jukebox 特征（data/audio_extraction/jukebox_features.py）
    # 按 AISTPPDataset.load_aistpp 的目录结构放入 data/train/ 或 data/test/

支持的输入格式（按扩展名自动识别）：
    .pkl  键 smpl_trans/smpl_poses（AIST++ 原始）或 trans/pose 或 pos/q；可选 keypoints3d/joints3d、fps、smpl_scaling
    .npy  F×75（前 3 列平移 + 72 轴角）或 F×72（平移置 0）
    .json dict 含 trans/pose/keypoints3d 数组，或逐帧 dict 列表（HybrIK 常见输出）
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np

RAW_FPS = 60  # EDGE 数据管线原始帧率（dataset.AISTPPDataset.raw_fps）

# 各关键点格式的脚部索引（OpenPose body25 / COCO-17），用于接触检测
FOOT_JOINTS = {
    "body25": {
        "lankle": [11], "rankle": [14],
        "ltoe": [19, 20, 21], "rtoe": [22, 23, 24],  # 大趾/小趾/脚跟
    },
    "coco17": {
        "lankle": [11], "rankle": [14],
        "ltoe": [11], "rtoe": [14],  # COCO 无足趾，退化为踝关节
    },
}

# z-up -> y-up：绕 X 轴 -90 度（EDGE 中 AIST++ y-up->z-up 用 RotateAxisAngle(90, "X") 的逆）
CONV_Z2Y = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


# ---------- 旋转工具（numpy 版 Rodrigues / 矩阵对数） ----------

def ax_to_mat(ax):
    """轴角 F×3 -> 旋转矩阵 F×3×3（Rodrigues）。"""
    ax = np.asarray(ax, dtype=np.float64)
    theta = np.linalg.norm(ax, axis=-1)
    k = ax / np.clip(theta[..., None], 1e-12, None)
    K = np.zeros((len(ax), 3, 3))
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]
    I = np.eye(3)[None]
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    R = I + sin_t[:, None, None] * K + (1.0 - cos_t)[:, None, None] * (K @ K)
    return R


def mat_to_ax(R):
    """旋转矩阵 F×3×3 -> 轴角 F×3（先 SVD 极投影正交化，再矩阵对数）。"""
    R = np.asarray(R, dtype=np.float64)
    U, _, Vt = np.linalg.svd(R)
    Rp = U @ Vt
    cos_t = np.clip((np.trace(Rp, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    rx = Rp[:, 2, 1] - Rp[:, 1, 2]
    ry = Rp[:, 0, 2] - Rp[:, 2, 0]
    rz = Rp[:, 1, 0] - Rp[:, 0, 1]
    sin_t = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2) / 2.0
    ax = np.stack([rx, ry, rz], axis=-1) * (theta / np.clip(2.0 * sin_t, 1e-12, None))[:, None]
    ax[theta < 1e-6] = 0.0
    return ax


# ---------- 平滑 ----------

def gaussian_smooth(x, sigma, axis=0):
    """沿指定轴的边缘填充高斯平滑（纯 numpy，边缘用最近邻复制保持长度）。"""
    x = np.asarray(x, dtype=np.float64)
    r = max(1, int(round(4.0 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    moved = np.moveaxis(x, axis, 0)
    padded = np.pad(moved, [(r, r)] + [(0, 0)] * (moved.ndim - 1), mode="edge")
    out = np.empty_like(moved)
    for i in range(moved.shape[0]):
        out[i] = np.tensordot(k, padded[i:i + 2 * r + 1], axes=(0, 0))
    return np.moveaxis(out, 0, axis)


def smooth_rotations(q, sigma=1.5):
    """轴角 F×72 平滑：转旋转矩阵逐元素高斯平滑后 SVD 投影回 SO(3)。"""
    q = np.asarray(q, dtype=np.float64).reshape(len(q), 24, 3)
    mats = ax_to_mat(q.reshape(-1, 3)).reshape(len(q), 24, 3, 3)
    smoothed = gaussian_smooth(mats.reshape(len(q), -1), sigma, axis=0)
    mats = smoothed.reshape(len(q), 24, 3, 3)
    return mat_to_ax(mats.reshape(-1, 3, 3)).reshape(len(q), 72)


def smooth_translation(trans, cutoff=6.0, fps=RAW_FPS, fallback_sigma=2.0):
    """位移低通滤波：优先 scipy Butterworth（零相位），无 scipy 时高斯平滑兜底。"""
    trans = np.asarray(trans, dtype=np.float64)
    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(2, cutoff / (fps / 2.0), btype="low")
        out = np.stack([filtfilt(b, a, trans[:, i]) for i in range(trans.shape[1])], axis=1)
        return out.astype(np.float64)
    except ImportError:
        return gaussian_smooth(trans, fallback_sigma, axis=0)


# ---------- 坐标系与重采样 ----------

def detect_up_axis(trans):
    """启发式判断竖直轴：竖直坐标恒为正（身高 ~1m），水平坐标均值趋近 0。"""
    means = np.asarray(trans, dtype=np.float64).mean(axis=0)
    if means[2] > 0.5 and means[2] > means[1]:
        return "z"
    return "y"


def to_y_up(trans, q, joints3d=None):
    """z-up -> y-up：平移与关键点旋转，SMPL 根关节(第 0 号)全局朝向左乘同一旋转。

    其余关节为相对父关节的局部旋转，不受世界坐标系变换影响。
    """
    trans = np.asarray(trans, dtype=np.float64) @ CONV_Z2Y.T
    q = np.asarray(q, dtype=np.float64).reshape(len(q), 24, 3).copy()
    root = ax_to_mat(q[:, 0])
    q[:, 0] = mat_to_ax(CONV_Z2Y @ root)
    if joints3d is not None:
        joints3d = np.asarray(joints3d, dtype=np.float64) @ CONV_Z2Y.T
    return trans, q.reshape(len(q), 72), joints3d


def resample_to_fps(trans, q, joints3d, fps_from, fps_to):
    """线性重采样到目标帧率：旋转在矩阵空间插值后 SVD 投影，平移逐轴 np.interp。"""
    fps_from, fps_to = float(fps_from), float(fps_to)
    if abs(fps_from - fps_to) < 1e-6:
        return trans, q, joints3d
    F = len(trans)
    n = int(round(F * fps_to / fps_from))
    t_old = np.arange(F) / fps_from
    t_new = np.arange(n) / fps_to
    trans = np.stack([np.interp(t_new, t_old, trans[:, i]) for i in range(3)], axis=1)
    mats = ax_to_mat(q.reshape(-1, 3)).reshape(F, 24, 3, 3).reshape(F, 216)
    new_flat = np.stack([np.interp(t_new, t_old, mats[:, i]) for i in range(216)], axis=1)
    q = mat_to_ax(new_flat.reshape(-1, 3, 3)).reshape(n, 72)
    if joints3d is not None:
        J = joints3d.shape[1]
        flat = joints3d.reshape(F, J * 3)
        new_flat = np.stack([np.interp(t_new, t_old, flat[:, i]) for i in range(J * 3)], axis=1)
        joints3d = new_flat.reshape(n, J, 3)
    return trans, q, joints3d


# ---------- 接触检测与脚步滑动修正 ----------

def mask_to_segments(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts, ends))


def drop_short_segments(mask, min_len):
    out = np.zeros_like(mask)
    for a, b in mask_to_segments(mask):
        if b - a + 1 >= min_len:
            out[a:b + 1] = 1
    return out


def detect_contacts(joints3d, fps, foot_def, v_thresh=0.01, height_thresh=0.12, min_len=3):
    """从 3D 关键点检测足部接触，输出 F×4 二值：[左脚踝, 右脚踝, 左脚, 右脚]。

    迟滞开关：速度 < 0.5*v_thresh 且高度 < 地面+height_thresh 时进入接触，
    速度 > 1.5*v_thresh 时离开；过短片段剔除。地面取各帧最低足高 1% 分位数。
    """
    joints3d = np.asarray(joints3d, dtype=np.float64)
    F = len(joints3d)
    all_kps = foot_def["lankle"] + foot_def["rankle"] + foot_def["ltoe"] + foot_def["rtoe"]
    feet_h = joints3d[:, all_kps][:, :, 1]
    ground = np.percentile(feet_h.min(axis=1), 1)

    def foot_vel(idx_list):
        p = joints3d[:, idx_list].mean(axis=1)
        v = np.linalg.norm(np.diff(p, axis=0), axis=-1)
        return np.concatenate([v[:1], v])  # v[t] = t 与 t+1 之间速度

    contacts = np.zeros((F, 4), dtype=np.float64)
    for c, idx_list in enumerate(
            [foot_def["lankle"], foot_def["rankle"], foot_def["ltoe"], foot_def["rtoe"]]):
        v = foot_vel(idx_list)
        h = joints3d[:, idx_list].mean(axis=1)[:, 1]
        state = False
        for t in range(F):
            if state:
                if v[t] > 1.5 * v_thresh:
                    state = False
            elif v[t] < 0.5 * v_thresh and h[t] < ground + height_thresh:
                state = True
            contacts[t, c] = float(state)
        contacts[:, c] = drop_short_segments(contacts[:, c], min_len)
    return contacts


def fix_foot_slide(trans, joints3d, contacts, foot_def, sigma=5.0):
    """用接触脚位置修正根平移漂移（只改水平 x/z，竖直 y 不动）。

    每个接触段取脚位置中位数作参考，把根平移向"让脚回到参考点"的方向修正，
    双脚同时接触时取平均，最后高斯平滑修正量。
    """
    trans = np.asarray(trans, dtype=np.float64)
    F = len(trans)
    corr = np.zeros((F, 3))
    w = np.zeros(F)
    for foot_i, (ankle_kps, toe_kps) in enumerate(
            [(foot_def["lankle"], foot_def["ltoe"]), (foot_def["rankle"], foot_def["rtoe"])]):
        mask = (contacts[:, foot_i] > 0.5) | (contacts[:, foot_i + 2] > 0.5)
        if mask.sum() < 2:
            continue
        foot_pos = joints3d[:, ankle_kps + toe_kps].mean(axis=1)
        for a, b in mask_to_segments(mask):
            ref = np.median(foot_pos[a:b + 1], axis=0)
            corr[a:b + 1] += (ref - foot_pos[a:b + 1]) * np.array([1.0, 0.0, 1.0])
            w[a:b + 1] += 1.0
    w = np.maximum(w, 1.0)
    corr = gaussian_smooth(corr / w[:, None], sigma, axis=0)
    corr[:, 1] = 0.0
    return trans + corr


# ---------- 输入加载 ----------

def load_estimator_file(path):
    """返回 (trans F×3, pose F×72, joints3d F×J×3|None, fps|None, scale)。

    支持 .pkl（AIST++ 原始键或 trans/pose）、.npy（F×75 或 F×72）、.json（数组或逐帧列表）。
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    trans = pose = joints3d = None
    fps = None
    scale = 1.0
    if ext == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
        for k in ("smpl_trans", "trans", "pos"):
            if k in data:
                trans = np.asarray(data[k], dtype=np.float64)
                break
        for k in ("smpl_poses", "pose", "q"):
            if k in data:
                pose = np.asarray(data[k], dtype=np.float64)
                break
        for k in ("keypoints3d", "joints3d", "joints"):
            if k in data:
                joints3d = np.asarray(data[k], dtype=np.float64)
                break
        fps = data.get("fps", data.get("fps_out"))
        if "smpl_scaling" in data:
            scale = float(np.asarray(data["smpl_scaling"]).reshape(-1)[0])
        elif "scale" in data:
            scale = float(np.asarray(data["scale"]).reshape(-1)[0])
    elif ext == ".npy":
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"{path}: 期望 2D 数组 (F×75 或 F×72)，得到 {arr.shape}")
        if arr.shape[1] == 75:
            trans, pose = arr[:, :3], arr[:, 3:]
        elif arr.shape[1] == 72:
            pose = arr
            print(f"[warn] {path}: 只有姿态无平移，trans 置 0")
        else:
            raise ValueError(f"{path}: 期望 75 或 72 列，得到 {arr.shape[1]}")
    elif ext == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k in ("trans", "transl", "pos"):
                if k in data:
                    trans = np.asarray(data[k], dtype=np.float64)
                    break
            for k in ("pose", "poses", "q"):
                if k in data:
                    pose = np.asarray(data[k], dtype=np.float64)
                    break
            for k in ("keypoints3d", "joints3d", "joints"):
                if k in data:
                    joints3d = np.asarray(data[k], dtype=np.float64)
                    break
            fps = data.get("fps")
        elif isinstance(data, list) and data:
            trans = np.asarray([f["trans"] for f in data], dtype=np.float64)
            pose = np.asarray([f["pose"] for f in data], dtype=np.float64)
            if "keypoints3d" in data[0]:
                joints3d = np.asarray([f["keypoints3d"] for f in data], dtype=np.float64)
        else:
            raise ValueError(f"{path}: 无法解析的 JSON 结构")
    else:
        raise ValueError(f"{path}: 不支持的扩展名（支持 .pkl/.npy/.json）")

    if pose is None:
        raise ValueError(f"{path}: 未找到姿态数据（键 pose/q/smpl_poses 之一）")
    pose = pose.reshape(len(pose), -1)
    if pose.shape[1] != 72:
        raise ValueError(f"{path}: 姿态维度应为 24×3=72，得到 {pose.shape[1]}")
    if trans is None:
        print(f"[warn] {path}: 未找到平移，trans 置 0")
        trans = np.zeros((len(pose), 3))
    trans = trans.reshape(len(pose), 3).astype(np.float64)
    if joints3d is not None:
        joints3d = joints3d.reshape(len(joints3d), -1, 3).astype(np.float64)
        assert len(joints3d) == len(pose), f"{path}: 关键点帧数 {len(joints3d)} 与姿态 {len(pose)} 不一致"
    return trans, pose, joints3d, fps, scale


# ---------- 输出与切窗 ----------

def write_sequence(path, trans, q, scale=1.0):
    with open(path, "wb") as f:
        pickle.dump({"pos": trans, "q": q, "scale": [scale]}, f)


def slice_sequence(trans, q, scale, name, out_dir, stride=0.5, length=5.0, num_slices=None):
    """与 data/slice.py slice_motion 相同约定：60fps，5s 窗 0.5s 步长，pos /= scale。"""
    trans = np.asarray(trans, dtype=np.float64) / scale
    q = np.asarray(q, dtype=np.float64)
    window = int(length * RAW_FPS)
    stride_step = int(stride * RAW_FPS)
    start_idx, count = 0, 0
    while start_idx <= len(trans) - window and (num_slices is None or count < num_slices):
        out = {"pos": trans[start_idx:start_idx + window], "q": q[start_idx:start_idx + window]}
        with open(os.path.join(out_dir, f"{name}_slice{count}.pkl"), "wb") as f:
            pickle.dump(out, f)
        start_idx += stride_step
        count += 1
    return count


def slice_wav(wav_file, out_dir, stride=0.5, length=5.0):
    """与 data/slice.py slice_audio 相同约定（需 librosa/soundfile）。"""
    import librosa
    import soundfile as sf
    audio, sr = librosa.load(wav_file, sr=None)
    name = os.path.splitext(os.path.basename(wav_file))[0]
    window = int(length * sr)
    stride_step = int(stride * sr)
    start_idx, count = 0, 0
    while start_idx <= len(audio) - window:
        sf.write(os.path.join(out_dir, f"{name}_slice{count}.wav"),
                 audio[start_idx:start_idx + window], sr)
        start_idx += stride_step
        count += 1
    return count


# ---------- 主流程 ----------

def clean_sequence(trans, q, joints3d, fps, opt):
    """完整清洗：坐标系 -> 重采样 -> 平滑 -> 接触 -> 脚步滑动修正。"""
    trans = np.asarray(trans, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    up = detect_up_axis(trans) if opt.up_axis == "auto" else opt.up_axis
    if up == "z":
        trans, q, joints3d = to_y_up(trans, q, joints3d)
        print(f"  坐标系: z-up -> y-up")
    trans, q, joints3d = resample_to_fps(trans, q, joints3d, float(fps), RAW_FPS)

    if not opt.no_smooth:
        q = smooth_rotations(q, opt.smooth_rot)
        trans = smooth_translation(trans, opt.smooth_trans, RAW_FPS)
        if joints3d is not None:
            joints3d = gaussian_smooth(joints3d, opt.smooth_rot, axis=0)

    contacts = None
    if joints3d is not None and opt.joints != "none":
        if opt.joints not in FOOT_JOINTS:
            raise ValueError(f"--joints 应为 {list(FOOT_JOINTS)} 之一，得到 {opt.joints}")
        contacts = detect_contacts(joints3d, RAW_FPS, FOOT_JOINTS[opt.joints],
                                   opt.contact_v_thresh, opt.contact_height, opt.min_contact_len)
        if opt.fix_foot_slide:
            trans = fix_foot_slide(trans, joints3d, contacts,
                                   FOOT_JOINTS[opt.joints], opt.fix_sigma)
    return {"trans": trans, "q": q, "joints3d": joints3d,
            "contacts": contacts, "fps": RAW_FPS}


def process_file(in_path, opt, out_root):
    name = os.path.splitext(os.path.basename(in_path))[0]
    trans, q, joints3d, fps, scale = load_estimator_file(in_path)
    fps = fps if fps else opt.input_fps
    print(f"{name}: {len(trans)} 帧 @ {fps}fps")
    if len(trans) < 30:
        print(f"[skip] {name}: 帧数过少")
        return

    data = clean_sequence(trans, q, joints3d, fps, opt)

    motions_dir = os.path.join(out_root, "motions")
    os.makedirs(motions_dir, exist_ok=True)
    write_sequence(os.path.join(motions_dir, f"{name}.pkl"),
                   data["trans"], data["q"], scale)

    if opt.save_contacts:
        if data["contacts"] is None:
            print(f"[warn] {name}: 无关键点输入，无法导出接触标签")
        else:
            contacts_dir = os.path.join(out_root, "contacts")
            os.makedirs(contacts_dir, exist_ok=True)
            np.save(os.path.join(contacts_dir, f"{name}_contacts.npy"), data["contacts"])

    if opt.slice:
        slice_dir = os.path.join(out_root, "motions_sliced")
        os.makedirs(slice_dir, exist_ok=True)
        num_slices = None
        if opt.wav_dir:
            wav_file = os.path.join(opt.wav_dir, f"{name}.wav")
            if os.path.isfile(wav_file):
                wav_slice_dir = os.path.join(out_root, "wavs_sliced")
                os.makedirs(wav_slice_dir, exist_ok=True)
                num_slices = slice_wav(wav_file, wav_slice_dir, opt.stride, opt.length)
                print(f"  音频切片 {num_slices} 段")
            else:
                print(f"[warn] {name}: --wav_dir 下无同名 wav，仅按动作长度切窗")
        count = slice_sequence(data["trans"], data["q"], scale, name, slice_dir,
                               opt.stride, opt.length, num_slices)
        print(f"  动作切片 {count} 段 (5s 窗 / {opt.stride}s 步长)")
        if num_slices is not None and count != num_slices:
            print(f"[warn] {name}: 音频 {num_slices} 段 != 动作 {count} 段（视频与音乐时长不一致？）")


def parse_opt():
    p = argparse.ArgumentParser(description="姿态估计输出清洗 -> EDGE 兼容 pkl（y-up 60fps，AIST++ 格式）")
    p.add_argument("--input_dir", type=str, required=True, help="姿态估计输出目录 (.pkl/.npy/.json)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="输出根目录，生成 motions/、motions_sliced/、wavs_sliced/、contacts/")
    p.add_argument("--up_axis", type=str, default="auto", choices=["auto", "y", "z"],
                   help="输入世界坐标竖直轴；auto 按根平移均值判断")
    p.add_argument("--input_fps", type=float, default=30.0, help="输入帧率（文件内 fps 键优先）")
    p.add_argument("--joints", type=str, default="none", choices=["none"] + list(FOOT_JOINTS),
                   help="3D 关键点格式，用于接触检测与脚步滑动修正")
    p.add_argument("--no_smooth", action="store_true", help="跳过平滑")
    p.add_argument("--smooth_rot", type=float, default=1.5, help="旋转高斯平滑 sigma（帧）")
    p.add_argument("--smooth_trans", type=float, default=6.0, help="位移 Butterworth 截止频率（Hz）")
    p.add_argument("--contact_v_thresh", type=float, default=0.01, help="接触速度阈值（米/帧 @60fps）")
    p.add_argument("--contact_height", type=float, default=0.12, help="接触高度阈值（地面以上米）")
    p.add_argument("--min_contact_len", type=int, default=3, help="最短接触段（帧），更短剔除")
    p.add_argument("--fix_foot_slide", action="store_true", help="按接触脚修正根平移漂移（需 --joints）")
    p.add_argument("--fix_sigma", type=float, default=5.0, help="脚步修正量平滑 sigma（帧）")
    p.add_argument("--save_contacts", action="store_true", help="导出接触标签 npy（F×4）")
    p.add_argument("--slice", action="store_true", help="切 5s/0.5s 训练窗")
    p.add_argument("--stride", type=float, default=0.5)
    p.add_argument("--length", type=float, default=5.0)
    p.add_argument("--wav_dir", type=str, default=None, help="同名 wav 目录，切窗时同步切音频")
    return p.parse_args()


def main():
    opt = parse_opt()
    files = []
    for ext in ("*.pkl", "*.npy", "*.json"):
        files += glob.glob(os.path.join(opt.input_dir, ext))
    files = sorted(set(files))
    if not files:
        print(f"{opt.input_dir} 下未找到 .pkl/.npy/.json 文件")
        return
    print(f"共 {len(files)} 个序列")
    for f in files:
        process_file(f, opt, opt.output_dir)
    print("完成")


if __name__ == "__main__":
    main()
