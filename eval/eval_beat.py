"""
Beat-alignment evaluation (module 6, musicality axis).

Measures how well motion onsets (peaks of wrist/foot velocity) align with
music beats: the mean distance from each motion onset to its nearest beat,
normalized by the beat period (lower = better, 0 = perfectly on beat).

Usage:
    python eval/eval_beat.py --motion_dir eval/motions/ --wav_dir custom_music/
    (matches name.pkl with name.wav)

Requires librosa and scipy (both optional: without librosa, beats are a fixed
grid at 120 BPM; without scipy, onsets use a local-max fallback).
"""

import argparse
import glob
import os
import pickle

import numpy as np

MOTION_JOINT_IDX = [7, 8, 10, 11, 20, 21]  # feet + wrists
FPS = 30


def beat_times(wav_path):
    try:
        import librosa

        y, sr = librosa.load(wav_path, sr=22050)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        return librosa.frames_to_time(beats, sr=sr), float(tempo)
    except ImportError:
        print("librosa not found; assuming a fixed 120 BPM grid")
        return np.arange(0, 60, 0.5), 120.0


def motion_onset_times(joint3d, fps=FPS, min_peak=0.02):
    rel = joint3d[:, MOTION_JOINT_IDX] - joint3d[:, 0:1]
    v = rel[1:] - rel[:-1]
    energy = np.linalg.norm(v, axis=-1).sum(axis=-1)  # (S-1,)
    energy = energy / (energy.max() + 1e-8)
    try:
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(energy, height=min_peak, distance=5)
    except ImportError:
        peaks = [
            i
            for i in range(1, len(energy) - 1)
            if energy[i] > min_peak
            and energy[i] >= energy[i - 1]
            and energy[i] >= energy[i + 1]
        ]
    return np.array(peaks) / fps


def beat_alignment_score(motion_onsets, beats, beat_period):
    if len(motion_onsets) == 0 or len(beats) == 0:
        return float("nan")
    dists = np.abs(motion_onsets[:, None] - beats[None, :]).min(axis=1)
    return float((dists / beat_period).mean())


def parse_eval_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_dir", type=str, default="motions/", help="Directory of saved .pkl motions"
    )
    parser.add_argument(
        "--wav_dir", type=str, default="custom_music/", help="Directory of matching .wav files"
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_eval_opt()
    scores = []
    for pkl in sorted(glob.glob(os.path.join(opt.motion_dir, "*.pkl"))):
        name = os.path.splitext(os.path.basename(pkl))[0]
        wav = os.path.join(opt.wav_dir, name + ".wav")
        if not os.path.isfile(wav):
            print(f"skip {name}: no matching wav")
            continue
        info = pickle.load(open(pkl, "rb"))
        onsets = motion_onset_times(info["full_pose"])
        beats, tempo = beat_times(wav)
        period = 60.0 / tempo
        score = beat_alignment_score(onsets, beats, period)
        scores.append(score)
        print(f"{name}: tempo {tempo:.1f} BPM, beat alignment {score:.4f}")
    if scores:
        print(f"mean beat alignment: {np.nanmean(scores):.4f}")
