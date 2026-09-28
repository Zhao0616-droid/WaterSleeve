"""
Style-fidelity evaluation (module 6).

1. Kinematic style features: translation-invariant, measurable proxies of the
   dance-professional attributes used for conditioning (劲道 / 水袖幅度 /
   重心起伏 ...), computable from any saved motion.
2. Intra/inter-style separation: distance between style clusters vs within
   them. A good style-controlled model should have small intra-style distance
   and large inter-style distance.
3. Style classifier scaffold: a small MLP trained on the features above, for
   quantitative style fidelity ("how often is the generated dance recognized
   as the requested style").

Usage:
    python eval/eval_style.py --motion_path eval/motions/               # one style
    python eval/eval_style.py --motion_path eval/motions_by_style/      # subdirs per style
    python eval/eval_style.py --motion_path eval/motions_by_style/ --train_classifier ckpt.pt
    python eval/eval_style.py --motion_path eval/motions_by_style/ --eval_classifier ckpt.pt
"""

import argparse
import glob
import os
import pickle

import numpy as np

WRIST_IDX = [20, 21]

FEATURE_NAMES = ["wrist_range", "wrist_med", "jerk_energy", "spine_ext", "arm_ext", "sway"]


def kinematic_style_features(joint3d):
    # joint3d: (S, 24, 3) FK positions
    root = joint3d[:, 0:1]
    rel = joint3d - root
    wrists = rel[:, WRIST_IDX]
    wrist_norm = wrists.norm(axis=-1)  # (S, 2)

    v = rel[1:] - rel[:-1]
    a = v[1:] - v[:-1]
    j = a[1:] - a[:-1]
    feats = np.array(
        [
            wrist_norm.max(),  # 水袖幅度 proxy
            wrist_norm.mean(),
            j.norm(axis=-1).mean(),  # 劲道 proxy
            np.linalg.norm(joint3d[:, 9] - joint3d[:, 6], axis=-1).mean(),  # 躯干伸展
            np.linalg.norm(joint3d[:, 22] - joint3d[:, 16], axis=-1).mean(),  # 手臂延展
            joint3d[:, 0, 2].std(),  # 重心起伏
        ]
    )
    return feats


def load_motions(dir_path):
    pkls = sorted(glob.glob(os.path.join(dir_path, "*.pkl")))
    feats, names = [], []
    for pkl in pkls:
        info = pickle.load(open(pkl, "rb"))
        feats.append(kinematic_style_features(info["full_pose"]))
        names.append(pkl)
    return np.array(feats), names


def print_style_summary(feats_by_style, style_names):
    print("\n=== style feature means (raw) ===")
    header = "style".ljust(12) + "".join(f.ljust(14) for f in FEATURE_NAMES)
    print(header)
    for name, feats in zip(style_names, feats_by_style):
        row = name.ljust(12) + "".join(f"{m:.4f}".ljust(14) for m in feats.mean(axis=0))
        print(row)

    print("\n=== intra/inter-style separation (normalized features, cosine) ===")
    all_feats = np.vstack(feats_by_style)
    mu, sigma = all_feats.mean(axis=0), all_feats.std(axis=0) + 1e-8
    normed = [(f - mu) / sigma for f in feats_by_style]
    centers = [f.mean(axis=0) for f in normed]
    print("pairwise inter-style distance matrix:")
    print("style".ljust(12) + "".join(n.ljust(12) for n in style_names))
    for i, n in enumerate(style_names):
        row = n.ljust(12)
        for j in range(len(style_names)):
            d = np.linalg.norm(centers[i] - centers[j])
            row += f"{d:.3f}".ljust(12)
        print(row)
    intra = [
        np.linalg.norm(f - c, axis=-1).mean() for f, c in zip(normed, centers)
    ]
    print("intra-style distance:", " ".join(f"{d:.3f}" for d in intra))


class StyleClassifier:
    def __init__(self, n_feats=len(FEATURE_NAMES), hidden=64):
        import torch

        self.torch = torch
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_feats, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def train(self, feats_by_style, epochs=300, lr=1e-3):
        torch = self.torch
        xs, ys = [], []
        for label, feats in enumerate(feats_by_style):
            xs.append(feats)
            ys.append(np.full(len(feats), label, dtype=np.float32))
        xs = torch.tensor(np.vstack(xs), dtype=torch.float32)
        ys = torch.tensor(np.concatenate(ys), dtype=torch.float32).unsqueeze(1)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(self.net(xs), ys)
            loss.backward()
            opt.step()
        return loss.item()

    def predict(self, feats):
        torch = self.torch
        with torch.no_grad():
            return self.net(torch.tensor(feats, dtype=torch.float32)).sigmoid().numpy()

    def save(self, path):
        self.torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(self.torch.load(path))


def parse_eval_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_path",
        type=str,
        default="motions/",
        help="Directory with .pkl motions, or a directory of per-style subdirectories",
    )
    parser.add_argument(
        "--train_classifier", type=str, default="", help="Train and save a style classifier to this path"
    )
    parser.add_argument(
        "--eval_classifier", type=str, default="", help="Evaluate a saved classifier on the motions"
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_eval_opt()
    subdirs = sorted(
        d
        for d in glob.glob(os.path.join(opt.motion_path, "*/"))
        if os.path.isdir(d)
    )
    if subdirs:
        style_names = [os.path.basename(d.rstrip("/\\")) for d in subdirs]
        feats_by_style = [load_motions(d)[0] for d in subdirs]
        print_style_summary(feats_by_style, style_names)
        if opt.train_classifier:
            clf = StyleClassifier()
            final_loss = clf.train(feats_by_style)
            clf.save(opt.train_classifier)
            print(f"trained classifier (final loss {final_loss:.4f}) -> {opt.train_classifier}")
        if opt.eval_classifier:
            clf = StyleClassifier()
            clf.load(opt.eval_classifier)
            for name, feats in zip(style_names, feats_by_style):
                preds = clf.predict(feats)
                acc = (preds > 0.5).astype(float).mean() if len(style_names) == 2 else None
                print(f"style '{name}': mean confidence {preds.mean():.3f}" + (f", binary acc {acc:.3f}" if acc is not None else ""))
    else:
        feats, names = load_motions(opt.motion_path)
        print_style_summary([feats], [opt.motion_path])
