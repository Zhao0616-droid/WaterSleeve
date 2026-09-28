"""data/pose_cleaning.py 的本地回归测试（纯 numpy，无 pytorch3d/scipy 依赖）。"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pose_cleaning import (CONV_Z2Y, detect_contacts, detect_up_axis,
                                fix_foot_slide, gaussian_smooth,
                                load_estimator_file, mask_to_segments,
                                mat_to_ax, ax_to_mat, process_file,
                                resample_to_fps, slice_sequence,
                                smooth_rotations, to_y_up)
from types import SimpleNamespace

RNG = np.random.default_rng(0)


def test_ax_mat_roundtrip():
    ax = RNG.uniform(-2.5, 2.5, size=(64, 3))
    err = np.abs(ax_to_mat(mat_to_ax(ax_to_mat(ax))) - ax_to_mat(ax)).max()
    assert err < 1e-6, err


def test_gaussian_smooth_reduces_jitter():
    t = np.arange(300)
    clean = 0.3 * np.sin(2 * np.pi * t / 120)
    noisy = clean + RNG.normal(0, 0.05, size=300)
    smoothed = gaussian_smooth(noisy[:, None], 2.0, axis=0)[:, 0]
    noise_in = np.std(noisy - clean)
    noise_out = np.std(smoothed - clean)
    assert noise_out < 0.4 * noise_in, (noise_in, noise_out)


def test_smooth_rotations_reduces_jitter():
    F = 200
    q = np.zeros((F, 72))
    q[:, 0:3] = [0, 0.5, 0]  # 恒定根旋转
    noisy = q + RNG.normal(0, 0.05, size=q.shape)
    smoothed = smooth_rotations(noisy, sigma=2.0)
    err_in = np.abs(noisy - q).mean()
    err_out = np.abs(smoothed - q).mean()
    assert err_out < 0.4 * err_in, (err_in, err_out)


def test_detect_up_axis():
    F = 100
    trans_y = np.zeros((F, 3)); trans_y[:, 1] = 0.95
    trans_z = np.zeros((F, 3)); trans_z[:, 2] = 0.95
    assert detect_up_axis(trans_y) == "y"
    assert detect_up_axis(trans_z) == "z"


def test_to_y_up():
    F = 10
    trans = np.zeros((F, 3)); trans[:, 2] = 1.0  # z-up 竖直
    q = np.zeros((F, 72))
    q[:, 0:3] = [0, 0, np.pi / 2]  # 绕 z 轴 90 度
    joints = np.zeros((F, 25, 3)); joints[:, :, 2] = 1.0
    trans2, q2, joints2 = to_y_up(trans, q, joints)
    assert np.allclose(trans2[:, 1], 1.0) and np.allclose(trans2[:, 2], 0.0)
    assert np.allclose(joints2[:, :, 1], 1.0)
    R_expected = CONV_Z2Y @ ax_to_mat(np.tile([0, 0, np.pi / 2], (F, 1)))
    R_got = ax_to_mat(q2[:, 0:3])
    assert np.allclose(R_got, R_expected, atol=1e-10)


def test_resample_to_fps():
    F, fps_from, fps_to = 150, 30.0, 60.0
    t = np.arange(F) / fps_from
    trans = np.stack([t, np.sin(t), np.zeros(F)], axis=1)
    q = np.zeros((F, 72))
    joints = np.zeros((F, 25, 3)); joints[:, 0, 0] = t
    trans2, q2, joints2 = resample_to_fps(trans, q, joints, fps_from, fps_to)
    assert len(trans2) == 300
    # 末帧超出原时间轴 1/60s 被 np.interp 截断，只校验前 299 帧
    assert np.allclose(trans2[:-1, 0], np.arange(299) / fps_to, atol=1e-9)
    assert np.allclose(joints2[:-1, 0, 0], np.arange(299) / fps_to, atol=1e-9)


def make_synthetic(frames=600, fps=60.0):
    """y-up 合成舞者：真实前进 0.5 m/s + 1.5Hz 抖动伪影（振幅 0.05，
    同时出现在根平移与脚关键点中——姿态估计的典型耦合伪影）；
    左脚周期性钉在地面（5s 接触 / 1s 跨步），右脚持续摆动。"""
    t = np.arange(frames) / fps
    jitter = 0.05 * np.sin(2 * np.pi * 1.5 * t)
    trans = np.stack([0.5 * t + jitter + RNG.normal(0, 0.003, frames),
                      0.95 + RNG.normal(0, 0.005, frames),
                      0.1 * np.sin(t)], axis=1)
    q = RNG.normal(0, 0.02, size=(frames, 72))  # 小扰动姿态
    joints = np.zeros((frames, 25, 3))
    joints[:, :, 1] = 0.9  # 其余关键点站立高度
    planted = (t % 6.0) < 5.0
    segs = mask_to_segments(planted)
    step = 0.5 * 6.0  # 每周期跨步 3m，跟上根前进
    x = np.full(frames, np.nan)
    for i, (a, b) in enumerate(segs):
        x[a:b + 1] = 0.3 + i * step
    for i in range(len(segs) - 1):  # 摆动段：上一接触点插值到下一接触点
        a, b = segs[i][1] + 1, segs[i + 1][0] - 1
        x[a:b + 1] = np.linspace(0.3 + i * step, 0.3 + (i + 1) * step, b - a + 1)
    if segs[0][0] > 0:
        x[:segs[0][0]] = np.linspace(0.3 - step, 0.3, segs[0][0])
    if segs[-1][1] < frames - 1:
        a = segs[-1][1] + 1
        x[a:] = np.linspace(0.3 + (len(segs) - 1) * step, 0.3 + len(segs) * step, frames - a)
    x += jitter + RNG.normal(0, 0.001, frames)
    y_planted = 0.04 + RNG.normal(0, 0.001, frames)
    # 左脚（11 踝 / 19,20,21 趾跟）
    for k in (11, 19, 20, 21):
        joints[:, k, 0] = x
        joints[:, k, 2] = 0.3
        joints[:, k, 1] = np.where(planted, y_planted, 0.5)
    # 右脚（14 踝 / 22,23,24 趾跟）：持续摆动，从不接触
    for k in (14, 22, 23, 24):
        joints[:, k, 0] = 0.3 + 0.2 * np.sin(2 * np.pi * t)
        joints[:, k, 2] = 0.3
        joints[:, k, 1] = 0.2 + 0.4 * np.abs(np.sin(np.pi * t))
    return trans, q, joints


def test_detect_contacts():
    trans, q, joints = make_synthetic()
    from data.pose_cleaning import FOOT_JOINTS
    contacts = detect_contacts(joints, 60.0, FOOT_JOINTS["body25"], v_thresh=0.02)
    planted = (np.arange(600) / 60.0) % 6.0 < 5.0
    assert contacts.shape == (600, 4)
    # 左脚踝通道应与 planted 基本一致（容忍边界帧与短段剔除）
    acc = (contacts[:, 0].astype(bool) == planted).mean()
    assert acc > 0.9, acc
    # 右脚持续摆动，不应有大段接触
    assert contacts[:, 1].sum() < 20, contacts[:, 1].sum()


def test_fix_foot_slide():
    trans, q, joints = make_synthetic()
    from data.pose_cleaning import FOOT_JOINTS
    contacts = detect_contacts(joints, 60.0, FOOT_JOINTS["body25"], v_thresh=0.02)
    fixed = fix_foot_slide(trans, joints, contacts, FOOT_JOINTS["body25"])
    planted = (np.arange(600) / 60.0) % 6.0 < 5.0
    segs = mask_to_segments(planted)

    def resid_from_true(tr):
        # 相对已知真实运动 0.5t 的偏差：抖动伪影留在里面，修正后应基本消除
        out = []
        for a, b in segs:
            tt = np.arange(a, b + 1) / 60.0
            out.append(tr[a:b + 1, 0] - 0.5 * tt)
        return np.concatenate(out)

    std_before = resid_from_true(trans).std()
    std_after = resid_from_true(fixed).std()
    assert std_before > 0.02, std_before  # 抖动伪影存在
    assert std_after < 0.3 * std_before, (std_before, std_after)
    # 竖直方向不动
    assert np.allclose(fixed[:, 1], trans[:, 1])


def test_load_estimator_file():
    trans, q, joints = make_synthetic(frames=120)
    with tempfile.TemporaryDirectory() as d:
        # pkl（AIST++ 原始键）
        import pickle
        with open(os.path.join(d, "a.pkl"), "wb") as f:
            pickle.dump({"smpl_trans": trans, "smpl_poses": q,
                         "smpl_scaling": [1.0], "fps": 60.0}, f)
        t1, q1, j1, fps1, s1 = load_estimator_file(os.path.join(d, "a.pkl"))
        assert np.allclose(t1, trans) and np.allclose(q1, q) and fps1 == 60.0 and s1 == 1.0
        # npy F×75
        np.save(os.path.join(d, "b.npy"), np.concatenate([trans, q], axis=1))
        t2, q2, j2, fps2, _ = load_estimator_file(os.path.join(d, "b.npy"))
        assert np.allclose(t2, trans) and np.allclose(q2, q) and fps2 is None
        # json 逐帧列表
        import json
        with open(os.path.join(d, "c.json"), "w") as f:
            json.dump([{"trans": trans[i].tolist(), "pose": q[i].tolist()}
                       for i in range(len(trans))], f)
        t3, q3, _, _, _ = load_estimator_file(os.path.join(d, "c.json"))
        assert np.allclose(t3, trans) and np.allclose(q3, q)


def test_process_file_slice():
    trans, q, joints = make_synthetic()
    opt = SimpleNamespace(up_axis="auto", input_fps=60.0, joints="body25",
                          no_smooth=False, smooth_rot=1.5, smooth_trans=6.0,
                          contact_v_thresh=0.02, contact_height=0.12, min_contact_len=3,
                          fix_foot_slide=True, fix_sigma=5.0,
                          save_contacts=True, slice=True, stride=0.5, length=5.0,
                          wav_dir=None)
    with tempfile.TemporaryDirectory() as d:
        raw_dir = os.path.join(d, "raw")
        os.makedirs(raw_dir)
        import pickle
        with open(os.path.join(raw_dir, "dance01.pkl"), "wb") as f:
            pickle.dump({"trans": trans, "pose": q, "keypoints3d": joints,
                         "fps": 60.0, "scale": [1.0]}, f)
        process_file(os.path.join(raw_dir, "dance01.pkl"), opt, os.path.join(d, "out"))

        with open(os.path.join(d, "out", "motions", "dance01.pkl"), "rb") as f:
            seq = pickle.load(f)
        assert seq["pos"].shape == (600, 3) and seq["q"].shape == (600, 72)
        assert seq["scale"] == [1.0]
        contacts = np.load(os.path.join(d, "out", "contacts", "dance01_contacts.npy"))
        assert contacts.shape == (600, 4)
        slices = sorted(os.listdir(os.path.join(d, "out", "motions_sliced")))
        assert len(slices) == 11, len(slices)  # 10s @60fps：0,30,...,300 → 11 窗
        with open(os.path.join(d, "out", "motions_sliced", slices[0]), "rb") as f:
            s0 = pickle.load(f)
        assert s0["pos"].shape == (300, 3) and s0["q"].shape == (300, 72)
        # 切窗应包含根平移修正后的内容（不再是纯线性漂移）
        assert not np.allclose(np.diff(s0["pos"][:, 0]), 0.5 / 60.0)


def test_slice_sequence_count():
    pos = np.zeros((600, 3))
    q = np.zeros((600, 72))
    with tempfile.TemporaryDirectory() as d:
        n = slice_sequence(pos, q, 1.0, "x", d, stride=0.5, length=5.0)
        assert n == 11


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
