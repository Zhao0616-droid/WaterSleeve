"""
Smoke test for the competition extensions (modules 2-4).

Runs without pytorch3d/vis/jukemirlib: those imports are replaced by minimal
but functionally correct stubs, so the conditioning, proxy-loss and
sequential-sampling plumbing can be verified on any machine with torch.

    python tests/smoke_test.py
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

# ---------------- stubs for optional heavy deps ----------------

_t3d = types.ModuleType("pytorch3d")
_t3dt = types.ModuleType("pytorch3d.transforms")


def axis_angle_to_matrix(aa):
    angle = aa.norm(dim=-1, keepdim=True) + 1e-8
    axis = aa / angle
    x, y, z = axis.unbind(-1)
    K = torch.zeros(aa.shape[:-1] + (3, 3), device=aa.device)
    K[..., 0, 1], K[..., 1, 0] = -z, z
    K[..., 0, 2], K[..., 2, 0] = y, -y
    K[..., 1, 2], K[..., 2, 1] = -x, x
    I = torch.eye(3, device=aa.device)
    return I + K * angle.sin() + K @ K * (1 - angle.cos())


def matrix_to_axis_angle(m):
    cos = ((m.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    angle = cos.acos()
    r00, r11, r22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    x = r22 - r11
    y = r00 - r22
    z = r11 - r00
    denom = (x**2 + y**2 + z**2).sqrt() + 1e-8
    axis = torch.stack([x, y, z], dim=-1) / denom.unsqueeze(-1)
    return axis * angle.unsqueeze(-1)


def rotation_6d_to_matrix(r6):
    a1, a2 = r6[..., :3], r6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rotation_6d(m):
    return torch.cat([m[..., :, 0], m[..., :, 1]], dim=-1)


def matrix_to_quaternion(m):
    # standard Shepperd's method (single branch is enough for a smoke test)
    t = m.diagonal(dim1=-2, dim2=-1).sum(-1)
    q = torch.zeros(m.shape[:-2] + (4,), device=m.device)
    q[..., 0] = (1 + t).clamp(min=0).sqrt() / 2
    q[..., 1] = (m[..., 2, 1] - m[..., 1, 2]).sign() * (1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2]).clamp(min=0).sqrt() / 2
    q[..., 2] = (m[..., 0, 2] - m[..., 2, 0]).sign() * (1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2]).clamp(min=0).sqrt() / 2
    q[..., 3] = (m[..., 1, 0] - m[..., 0, 1]).sign() * (1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]).clamp(min=0).sqrt() / 2
    return q


def quaternion_to_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)], -1),
        ],
        -1,
    )


_t3dt.axis_angle_to_matrix = axis_angle_to_matrix
_t3dt.matrix_to_axis_angle = matrix_to_axis_angle
_t3dt.rotation_6d_to_matrix = rotation_6d_to_matrix
_t3dt.matrix_to_rotation_6d = matrix_to_rotation_6d
_t3dt.matrix_to_quaternion = matrix_to_quaternion
_t3dt.quaternion_to_matrix = quaternion_to_matrix
_t3dt.axis_angle_to_quaternion = lambda aa: matrix_to_quaternion(axis_angle_to_matrix(aa))
_t3dt.quaternion_to_axis_angle = lambda q: matrix_to_axis_angle(quaternion_to_matrix(q))
_t3d.transforms = _t3dt
sys.modules["pytorch3d"] = _t3d
sys.modules["pytorch3d.transforms"] = _t3dt

_vis = types.ModuleType("vis")
_vis.skeleton_render = lambda *a, **k: None
sys.modules["vis"] = _vis

_ptqdm = types.ModuleType("p_tqdm")
_ptqdm.p_map = lambda f, it: list(map(f, it))
sys.modules["p_tqdm"] = _ptqdm

# ---------------- imports under test ----------------

from model.model import DanceDecoder  # noqa: E402
from model.diffusion import GaussianDiffusion, compute_motion_attrs  # noqa: E402


class FakeSMPL:
    def forward(self, q, x):
        # deterministic FK stand-in: positions depend on both root and joints
        return x[:, :, None, :3] + 0.1 * q


def make_decoder(num_styles=0, attr_dim=0, latent_dim=64, seq_len=16):
    return DanceDecoder(
        nfeats=151,
        seq_len=seq_len,
        latent_dim=latent_dim,
        ff_size=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        cond_feature_dim=32,
        num_styles=num_styles,
        attr_dim=attr_dim,
    )


def test_decoder_conditioning():
    print("[1] DanceDecoder conditioning")
    B, S = 2, 16
    x = torch.randn(B, S, 151)
    cond = torch.randn(B, S, 32)
    t = torch.randint(0, 100, (B,)).long()

    # backward compatibility: no style/attr heads
    m0 = make_decoder(num_styles=0, attr_dim=0)
    out0 = m0(x, cond, t, cond_drop_prob=0.1)
    assert out0.shape == (B, S, 151)
    g0 = m0.guided_forward(x, cond, t, 2.0)
    assert g0.shape == out0.shape

    # style + attr heads
    m = make_decoder(num_styles=3, attr_dim=2)
    style_ids = torch.tensor([0, 2]).long()
    attrs = torch.randn(B, 2)

    for sd, ad in [(0.0, 0.0), (1.0, 1.0)]:
        out = m(x, cond, t, cond_drop_prob=0.1, style_ids=style_ids,
                style_drop_prob=sd, attrs=attrs, attr_drop_prob=ad)
        assert out.shape == (B, S, 151), out.shape

    # no style/attrs passed -> null embeddings, must not crash
    out = m(x, cond, t, cond_drop_prob=0.0)
    assert out.shape == (B, S, 151)

    # negative style ids -> explicit unconditional
    out = m(x, cond, t, cond_drop_prob=0.0, style_ids=torch.tensor([-1, -1]).long())
    assert out.shape == (B, S, 151)

    # multi-condition guidance: 3 passes (music, style, attrs)
    g = m.guided_forward(x, cond, t, 2.0, style_ids=style_ids,
                         style_guidance_weight=1.5, attrs=attrs,
                         attr_guidance_weight=0.5)
    assert g.shape == (B, S, 151)

    # different conditions produce different outputs
    out_a = m(x, cond, t, style_ids=torch.tensor([0, 0]).long(), attrs=attrs)
    out_b = m(x, cond, t, style_ids=torch.tensor([1, 1]).long(), attrs=attrs)
    assert not torch.allclose(out_a, out_b)
    print("    OK")


def test_proxy_losses():
    print("[2] p_losses with costume physics proxies + self-supervised attrs")
    B, S = 2, 16
    decoder = make_decoder(num_styles=2, attr_dim=3)
    diffusion = GaussianDiffusion(
        decoder, S, 151, FakeSMPL(),
        n_timestep=100, predict_epsilon=False, loss_type="l2",
        attr_names=["energy", "sleeve_amp", "com_sway"],
        wrist_smooth_weight=1.0, arc_weight=1.0, penetration_weight=1.0,
        penetration_thresh=0.05,
    )
    x = torch.randn(B, S, 151)
    cond = torch.randn(B, S, 32)
    style_ids = torch.tensor([0, 1]).long()

    # with attrs given
    attrs = torch.randn(B, 3)
    total, losses = diffusion.p_losses(x, cond, torch.randint(0, 100, (B,)).long(),
                                       style_ids=style_ids, attrs=attrs)
    assert total.ndim == 0
    assert len(losses) == 7, f"expected 4 base + 3 proxy losses, got {len(losses)}"
    assert all(l.ndim == 0 for l in losses)

    # self-supervised attrs from ground-truth FK
    total2, losses2 = diffusion.p_losses(
        x, cond, torch.randint(0, 100, (B,)).long(), style_ids=None, attrs=None
    )
    assert total2.ndim == 0 and len(losses2) == 7

    # zero weights -> original 4-loss behavior
    diffusion.wrist_smooth_weight = 0.0
    diffusion.arc_weight = 0.0
    diffusion.penetration_weight = 0.0
    _, losses3 = diffusion.p_losses(x, cond, torch.randint(0, 100, (B,)).long())
    assert len(losses3) == 4
    print("    OK")


def test_compute_motion_attrs():
    print("[3] compute_motion_attrs")
    xp = torch.randn(2, 16, 24, 3)
    attrs = compute_motion_attrs(xp, ["energy", "sleeve_amp", "com_sway"])
    assert attrs.shape == (2, 3)
    # translation invariance
    attrs2 = compute_motion_attrs(xp + 5.0, ["energy", "sleeve_amp", "com_sway"])
    assert torch.allclose(attrs, attrs2, atol=1e-5)
    print("    OK")


def test_sequential_sample():
    print("[4] sequential_sample cross-window consistency")
    B, S = 2, 16
    overlap = 4
    decoder = make_decoder(num_styles=0, attr_dim=0)
    diffusion = GaussianDiffusion(
        decoder, S, 151, FakeSMPL(),
        n_timestep=100, predict_epsilon=False, loss_type="l2",
    )
    cond = torch.randn(B, S, 32)
    out = diffusion.sequential_sample((B, S, 151), cond, overlap=overlap)
    assert out.shape == (B, S, 151)
    # window i+1's first `overlap` frames must match window i's tail exactly
    diff = (out[1, :overlap] - out[0, -overlap:]).abs().max().item()
    assert diff < 1e-4, f"overlap mismatch: {diff}"
    # style/attr per-window slicing must not crash
    out2 = diffusion.sequential_sample(
        (B, S, 151), cond, overlap=overlap,
        style_ids=torch.tensor([0, 0]).long(), attrs=torch.zeros(B, 1),
        guidance_schedule=[1.0, 2.0],
    )
    assert out2.shape == (B, S, 151)
    print("    OK")


if __name__ == "__main__":
    test_decoder_conditioning()
    test_proxy_losses()
    test_compute_motion_attrs()
    test_sequential_sample()
    print("ALL SMOKE TESTS PASSED")
