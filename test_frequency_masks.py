import torch
import numpy as np

from function.dct import dct_2d, idct_2d


def make_frequency_masks(H, W, low=0.20, mid=0.50, device="cuda"):
    fy = torch.arange(H, device=device, dtype=torch.float32)
    fx = torch.arange(W, device=device, dtype=torch.float32)

    fy = fy.view(H, 1) / max(H - 1, 1)
    fx = fx.view(1, W) / max(W - 1, 1)

    radius = torch.sqrt(fy ** 2 + fx ** 2) / np.sqrt(2.0)

    mask_lf = radius <= low
    mask_mf = (radius > low) & (radius <= mid)
    mask_hf = radius > mid

    return (
        mask_lf[None, None],
        mask_mf[None, None],
        mask_hf[None, None],
    )


def test(H, W):
    torch.manual_seed(123)

    x = torch.randn(
        1, 256 if H == 35 else 768, H, W,
        device="cuda",
        requires_grad=True,
    )

    X = dct_2d(x, norm="ortho")

    lf_mask, mf_mask, hf_mask = make_frequency_masks(H, W)

    X_lf = X * lf_mask
    X_mf = X * mf_mask
    X_hf = X * hf_mask

    x_lf = idct_2d(X_lf, norm="ortho")
    x_mf = idct_2d(X_mf, norm="ortho")
    x_hf = idct_2d(X_hf, norm="ortho")

    x_rec = x_lf + x_mf + x_hf

    reconstruction_error = (x_rec - x).abs().max()

    energy = (X ** 2).sum()

    energy_lf = (X_lf ** 2).sum() / energy
    energy_mf = (X_mf ** 2).sum() / energy
    energy_hf = (X_hf ** 2).sum() / energy

    loss = (
        x_lf.square().mean()
        + x_mf.square().mean()
        + x_hf.square().mean()
    )

    loss.backward()

    print(f"\nFeature: {H}x{W}")
    print(f"LF coefficients: {lf_mask.sum().item()}")
    print(f"MF coefficients: {mf_mask.sum().item()}")
    print(f"HF coefficients: {hf_mask.sum().item()}")
    print(f"Reconstruction error: {reconstruction_error.item():.8e}")
    print(f"LF energy: {energy_lf.item():.6f}")
    print(f"MF energy: {energy_mf.item():.6f}")
    print(f"HF energy: {energy_hf.item():.6f}")
    print(f"Energy sum: {(energy_lf + energy_mf + energy_hf).item():.6f}")
    print(f"Gradient finite: {torch.isfinite(x.grad).all().item()}")


if __name__ == "__main__":
    test(35, 35)
    test(17, 17)
