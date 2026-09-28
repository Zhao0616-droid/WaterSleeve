"""
Music-structure helpers for long-form generation (module 4).

Derives per-window classifier-free guidance weights from onset density so that
energetic sections (e.g. choruses with dense beats) are guided more strongly
than sparse sections (e.g. intros). librosa is optional: without it a uniform
schedule is returned.
"""

import numpy as np


def _onset_density(wav_path, sr=22050):
    import librosa

    y, sr = librosa.load(wav_path, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    return float(onset_env.mean()) if len(onset_env) else 0.0


def window_guidance_schedule(
    wav_files, base_guidance=2.0, low=0.8, high=1.2
):
    """
    wav_files: one (possibly sliced) wav per generation window, in order.
    Returns per-window guidance weights relative to the mean onset density.
    """
    try:
        import librosa  # noqa: F401
    except ImportError:
        print("librosa not found; using uniform guidance schedule")
        return [base_guidance] * len(wav_files)

    densities = np.array([_onset_density(f) for f in wav_files])
    rel = densities / (densities.mean() + 1e-8)
    return [base_guidance * float(np.clip(d, low, high)) for d in rel]
