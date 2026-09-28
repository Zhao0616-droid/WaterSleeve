import argparse
import glob
import os
import pickle
import random

import numpy as np
from tqdm import tqdm

WRIST_IDX = [20, 21]
ELBOW_IDX = [18, 19]
SPINE_IDX = 6
CHEST_IDX = 9


def calc_physical_score(dir):
    scores = []
    names = []
    accelerations = []
    up_dir = 2  # z is up
    flat_dirs = [i for i in range(3) if i != up_dir]
    DT = 1 / 30

    it = glob.glob(os.path.join(dir, "*.pkl"))
    if len(it) > 1000:
        it = random.sample(it, 1000)
    for pkl in tqdm(it):
        info = pickle.load(open(pkl, "rb"))
        joint3d = info["full_pose"]
        root_v = (joint3d[1:, 0, :] - joint3d[:-1, 0, :]) / DT  # root velocity (S-1, 3)
        root_a = (root_v[1:] - root_v[:-1]) / DT  # (S-2, 3) root accelerations
        # clamp the up-direction of root acceleration
        root_a[:, up_dir] = np.maximum(root_a[:, up_dir], 0)  # (S-2, 3)
        # l2 norm
        root_a = np.linalg.norm(root_a, axis=-1)  # (S-2,)
        scaling = root_a.max()
        root_a /= scaling

        foot_idx = [7, 10, 8, 11]
        feet = joint3d[:, foot_idx]  # foot positions (S, 4, 3)
        foot_v = np.linalg.norm(
            feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1
        )  # (S-2, 4) horizontal velocity
        foot_mins = np.zeros((len(foot_v), 2))
        foot_mins[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1])
        foot_mins[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3])

        foot_loss = (
            foot_mins[:, 0] * foot_mins[:, 1] * root_a
        )  # min leftv * min rightv * root_a (S-2,)
        foot_loss = foot_loss.mean()
        scores.append(foot_loss)
        names.append(pkl)
        accelerations.append(foot_mins[:, 0].mean())

    out = np.mean(scores) * 10000
    print(f"{dir} has a mean PFC of {out}")


def calc_cloth_physics(dir_path, penetration_thresh=0.05):
    """
    Costume physics proxies (module 3 / module 6), mirroring the training-time
    losses so they can be reported as extended physical-plausibility metrics:
    - wrist smoothness: mean wrist jerk relative to root (lower = smoother)
    - arc smoothness: mean direction change of wrist velocity (lower = more
      circular/flowing sleeve-like trajectories)
    - penetration: fraction of elbow/wrist positions closer than `thresh` to
      the chest-spine line (lower = fewer arm-torso interpenetrations)
    """
    wrist_jerks, arc_changes, pen_fracs = [], [], []
    for pkl in tqdm(glob.glob(os.path.join(dir_path, "*.pkl"))):
        info = pickle.load(open(pkl, "rb"))
        xp = info["full_pose"]  # (S, 24, 3)
        rel = xp[:, WRIST_IDX] - xp[:, 0:1]
        v = rel[1:] - rel[:-1]
        a = v[1:] - v[:-1]
        j = a[1:] - a[:-1]
        wrist_jerks.append(np.linalg.norm(j, axis=-1).mean())

        vn = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
        arc_changes.append((1 - (vn[1:] * vn[:-1]).sum(-1)).mean())

        seg = xp[:, ELBOW_IDX + WRIST_IDX]
        spine = xp[:, SPINE_IDX : SPINE_IDX + 1]
        chest = xp[:, CHEST_IDX : CHEST_IDX + 1]
        line = chest - spine
        line_len2 = (line**2).sum(-1, keepdims=True) + 1e-8
        tt = ((seg - spine) * line).sum(-1, keepdims=True) / line_len2
        closest = spine + np.clip(tt, 0, 1) * line
        dist = np.linalg.norm(seg - closest, axis=-1)
        pen_fracs.append((dist < penetration_thresh).mean())

    print(f"{dir_path}:")
    print(f"  wrist smoothness (jerk):   {np.mean(wrist_jerks):.6f}")
    print(f"  arc smoothness (dir chg):  {np.mean(arc_changes):.6f}")
    print(f"  penetration fraction:      {np.mean(pen_fracs):.6f}")


def parse_eval_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_path",
        type=str,
        default="motions/",
        help="Where to load saved motions",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pfc", "cloth", "all"],
        default="pfc",
        help="pfc: foot contact only; cloth: costume physics proxies; all: both",
    )
    parser.add_argument(
        "--penetration_thresh",
        type=float,
        default=0.05,
        help="Distance threshold for the arm-torso penetration fraction",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_eval_opt()
    if opt.mode in ("pfc", "all"):
        calc_physical_score(opt.motion_path)
    if opt.mode in ("cloth", "all"):
        calc_cloth_physics(opt.motion_path, opt.penetration_thresh)
