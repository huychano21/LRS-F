import os
import sys
import csv
import argparse
import importlib.util
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader
import pretrainedmodels

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from function.loader import ImageNet
from function.Normalize import Normalize
from function.LRS import (
    multi_lrsf_inv3,
    calculate_lrs_parameters,
    altern_ls,
    inv3_logit,
)
from function.dct import dct_2d, idct_2d


# Load the exact Step-8 MI-LRSF implementation so that
# DCT/masks/frequency_objective are identical to the attack code.
MI_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "MI-LRSF.py"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "MI-LRSF(2).py"),
]
MI_PATH = next((p for p in MI_CANDIDATES if os.path.exists(p)), None)
if MI_PATH is None:
    raise FileNotFoundError(
        "Could not find MI-LRSF.py or MI-LRSF(2).py in "
        + os.path.dirname(os.path.abspath(__file__))
    )
spec = importlib.util.spec_from_file_location("mi_lrsf_step8", MI_PATH)
mi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mi)


def keep_frequency(feature, band, lf_threshold=0.20, mf_threshold=0.50):
    coeff = dct_2d(feature)
    _, _, h, w = feature.shape

    yy, xx = torch.meshgrid(
        torch.arange(h, device=feature.device, dtype=feature.dtype),
        torch.arange(w, device=feature.device, dtype=feature.dtype),
        indexing="ij",
    )

    yy = yy / max(h - 1, 1)
    xx = xx / max(w - 1, 1)
    radius = torch.sqrt(yy * yy + xx * xx) / np.sqrt(2.0)

    if band == "L":
        mask = radius <= lf_threshold
    elif band == "M":
        mask = (radius > lf_threshold) & (radius <= mf_threshold)
    elif band == "H":
        mask = radius > mf_threshold
    else:
        raise ValueError("Unknown band: " + str(band))

    mask = mask.to(coeff.dtype).view(1, 1, h, w)
    return idct_2d(coeff * mask), mask, coeff


def forward_prefix(model, inp):
    x = model.Conv2d_1a_3x3(inp)
    x = model.Conv2d_2a_3x3(x)
    x = model.Conv2d_2b_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = model.Conv2d_3b_1x1(x)
    x = model.Conv2d_4a_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    return x


def debug_lrsf_inv3(
    model,
    inp,
    num_iters=5,
    crs=0.8,
    rrs=0.01,
    crm=0.5,
    rrm=0.04,
    crd=0.0,
    rrd=0.1,
    lf=0.20,
    mf=0.50,
):
    """
    Reconstruct the exact Inception-v3 LRS-F graph while exposing
    the intermediate expert logits and feature components.

    This mirrors function.LRS.multi_lrsf_inv3().
    """

    # ------------------------------------------------------------
    # SHALLOW: Sparse + High Frequency
    # ------------------------------------------------------------
    xs = model.Mixed_5b(forward_prefix(model, inp))
    b, c, h, w = xs.shape

    rank, k = calculate_lrs_parameters(c, h * w, crs, rrs)
    feat = xs.view(b, c, h * w).float()

    ds = torch.clamp(
        torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True)),
        min=1e-8,
    )
    norm = feat / ds

    _, sparse = altern_ls(
        norm,
        num_iters,
        target_rank=rank,
        num_nonzeros=k,
    )
    sparse = (sparse * ds).view(b, c, h, w)

    fh, mask_s, _ = keep_frequency(xs, "H", lf, mf)
    rs = sparse + fh

    ys = model.Mixed_5c(rs)
    ys = model.Mixed_5d(ys)
    ys = model.Mixed_6a(ys)
    ys = model.Mixed_6b(ys)
    ys = model.Mixed_6c(ys)
    ys = model.Mixed_6d(ys)
    ys = model.Mixed_6e(ys)
    z_s = inv3_logit(model, ys)

    # ------------------------------------------------------------
    # MIDDLE: (Low-Rank + Sparse)/2 + Mid Frequency
    # ------------------------------------------------------------
    xm = model.Mixed_5b(forward_prefix(model, inp))
    xm = model.Mixed_5c(xm)
    xm = model.Mixed_5d(xm)

    b, c, h, w = xm.shape
    rank, k = calculate_lrs_parameters(c, h * w, crm, rrm)
    feat = xm.view(b, c, h * w).float()

    dm = torch.clamp(
        torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True)),
        min=1e-8,
    )
    norm = feat / dm

    low_m, sparse_m = altern_ls(
        norm,
        num_iters,
        target_rank=rank,
        num_nonzeros=k,
    )

    lrs_m = (low_m + sparse_m) * dm / 2.0
    lrs_m = lrs_m.view(b, c, h, w)

    fm, mask_m, _ = keep_frequency(xm, "M", lf, mf)
    rm = lrs_m + fm

    ym = model.Mixed_6a(rm)
    ym = model.Mixed_6b(ym)
    ym = model.Mixed_6c(ym)
    ym = model.Mixed_6d(ym)
    ym = model.Mixed_6e(ym)
    z_m = inv3_logit(model, ym)

    # ------------------------------------------------------------
    # DEEP: Low-Rank + Low Frequency
    # ------------------------------------------------------------
    xd = forward_prefix(model, inp)
    xd = model.Mixed_5b(xd)
    xd = model.Mixed_5c(xd)
    xd = model.Mixed_5d(xd)
    xd = model.Mixed_6a(xd)
    xd = model.Mixed_6b(xd)
    xd = model.Mixed_6c(xd)
    xd = model.Mixed_6d(xd)
    xd = model.Mixed_6e(xd)

    b, c, h, w = xd.shape
    rank, k = calculate_lrs_parameters(c, h * w, crd, rrd)
    feat = xd.view(b, c, h * w).float()

    dd = torch.clamp(
        torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True)),
        min=1e-8,
    )
    norm = feat / dd

    low_d, _ = altern_ls(
        norm,
        num_iters,
        target_rank=rank,
        num_nonzeros=k,
    )
    low_d = (low_d * dd).view(b, c, h, w)

    fl, mask_d, _ = keep_frequency(xd, "L", lf, mf)
    rd = low_d + fl
    z_d = inv3_logit(model, rd)

    # ------------------------------------------------------------
    # ORIGINAL
    # ------------------------------------------------------------
    xo = forward_prefix(model, inp)
    xo = model.Mixed_5b(xo)
    xo = model.Mixed_5c(xo)
    xo = model.Mixed_5d(xo)
    xo = model.Mixed_6a(xo)
    xo = model.Mixed_6b(xo)
    xo = model.Mixed_6c(xo)
    xo = model.Mixed_6d(xo)
    xo = model.Mixed_6e(xo)
    z_o = inv3_logit(model, xo)

    z_fused = (z_s + z_m + z_d + z_o) / 4.0

    return {
        "zS": z_s,
        "zM": z_m,
        "zD": z_d,
        "zO": z_o,
        "zf": z_fused,
        "xs": xs,
        "xm": xm,
        "xd": xd,
        "S": sparse,
        "Lm": low_m * dm,
        "Sm": sparse_m * dm,
        "Ld": low_d,
        "FH": fh,
        "FM": fm,
        "FL": fl,
        "Rs": rs,
        "Rm": rm,
        "Rd": rd,
        "masks": (mask_s, mask_m, mask_d),
    }


def band_energy_ratios(coeff, masks):
    energy = coeff * coeff
    total = energy.sum(dim=(1, 2, 3)) + 1e-12
    result = []
    for mask in masks:
        value = (
            (energy * mask).sum(dim=(1, 2, 3)) / total
        ).mean().item()
        result.append(value)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Step 9A mechanistic validation for LRS-F."
    )

    parser.add_argument(
        "--input_csv",
        default=os.path.join(PROJECT_ROOT, "dataset/test10.csv"),
    )
    parser.add_argument(
        "--input_dir",
        default=os.path.join(PROJECT_ROOT, "dataset/images"),
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Number of images used for the mechanistic test.",
    )
    parser.add_argument("--lrs_num_iters", type=int, default=5)
    parser.add_argument("--lf_threshold", type=float, default=0.20)
    parser.add_argument("--mf_threshold", type=float, default=0.50)
    parser.add_argument("--eps", type=float, default=16.0 / 255.0)
    parser.add_argument(
        "--output_csv",
        default=os.path.join(
            PROJECT_ROOT,
            "Attack/outputs/step9A_mechanistic_validation.csv",
        ),
    )

    args = parser.parse_args()

    torch.manual_seed(12345)
    np.random.seed(12345)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(12345)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
    ])

    dataset = ImageNet(
        args.input_dir,
        args.input_csv,
        transforms=transform,
    )

    n = min(args.num_images, len(dataset))
    dataset = torch.utils.data.Subset(
        dataset,
        list(range(n)),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    normalize = Normalize(
        np.array([0.5, 0.5, 0.5]),
        np.array([0.5, 0.5, 0.5]),
    ).to(device)

    print("=" * 70)
    print("STEP 9A - MECHANISTIC VALIDATION")
    print("=" * 70)
    print("Device:", device)
    print("Images:", len(dataset))
    print("LF threshold:", args.lf_threshold)
    print("MF threshold:", args.mf_threshold)
    print("LRS iterations:", args.lrs_num_iters)
    print()

    print("Loading pretrainedmodels Inception-v3...")
    model = pretrainedmodels.inceptionv3(
        num_classes=1000,
        pretrained="imagenet",
    ).to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    rows = []

    for batch_idx, (images, names, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        x = normalize(images)

        # Reset RNG before both implementations because altern_ls()
        # contains random initialization. This makes the comparison
        # between the debug graph and production function deterministic.
        torch.manual_seed(12345 + batch_idx)

        with torch.no_grad():
            debug = debug_lrsf_inv3(
                model,
                x,
                num_iters=args.lrs_num_iters,
                lf=args.lf_threshold,
                mf=args.mf_threshold,
            )

        torch.manual_seed(12345 + batch_idx)

        with torch.no_grad():
            production = multi_lrsf_inv3(
                model,
                x,
                num_iters=args.lrs_num_iters,
                compression_rate_shallow=0.8,
                rank_ratio_shallow=0.01,
                compression_rate_balanced=0.5,
                rank_ratio_balanced=0.04,
                compression_rate_deep=0.0,
                rank_ratio_deep=0.1,
                lf_threshold=args.lf_threshold,
                mf_threshold=args.mf_threshold,
            )

        fusion_error = (
            debug["zf"] - production
        ).abs().max().item()

        # --------------------------------------------------------
        # Frequency-mask partition check
        # --------------------------------------------------------
        # The three production masks are at different feature
        # resolutions (35x35, 35x35, 17x17), so they must NOT be
        # added together. Check LF/MF/HF partition independently
        # at each spatial resolution.
        partition_errors = []
        overlap_errors = []

        for feature in [debug["xs"], debug["xd"]]:
            h, w = feature.shape[-2:]
            masks = mi.make_frequency_masks(
                h,
                w,
                args.lf_threshold,
                args.mf_threshold,
                feature.device,
                feature.dtype,
            )

            lf_mask, mf_mask, hf_mask = masks

            mask_sum = lf_mask + mf_mask + hf_mask
            partition_errors.append(
                (mask_sum - 1.0).abs().max().item()
            )

            overlap_errors.append(
                max(
                    (lf_mask * mf_mask).abs().max().item(),
                    (lf_mask * hf_mask).abs().max().item(),
                    (mf_mask * hf_mask).abs().max().item(),
                )
            )

        partition_error = max(partition_errors)
        overlap_error = max(overlap_errors)

        # --------------------------------------------------------
        # DCT/IDCT reconstruction
        # --------------------------------------------------------
        recon_errors = []

        for feature in [
            debug["xs"],
            debug["xm"],
            debug["xd"],
        ]:
            coeff = dct_2d(feature)
            reconstructed = idct_2d(coeff)
            recon_errors.append(
                (reconstructed - feature).abs().max().item()
            )

        dct_reconstruction_error = max(recon_errors)

        # --------------------------------------------------------
        # Frequency isolation checks
        # --------------------------------------------------------
        frequency_ratios = []

        for feature in [
            debug["FH"],
            debug["FM"],
            debug["FL"],
        ]:
            coeff = dct_2d(feature)

            masks = mi.make_frequency_masks(
                feature.shape[-2],
                feature.shape[-1],
                args.lf_threshold,
                args.mf_threshold,
                feature.device,
                feature.dtype,
            )

            frequency_ratios.append(
                band_energy_ratios(coeff, masks)
            )

        shallow_h = frequency_ratios[0][2]
        middle_m = frequency_ratios[1][1]
        deep_l = frequency_ratios[2][0]

        # --------------------------------------------------------
        # Expert logits and fused logits
        # --------------------------------------------------------
        z_o = debug["zO"]
        z_s = debug["zS"]
        z_m = debug["zM"]
        z_d = debug["zD"]
        z_f = debug["zf"]

        ce_loss = F.cross_entropy(
            z_f,
            labels,
        ).item()

        predictions = {
            "O": z_o.argmax(dim=1),
            "S": z_s.argmax(dim=1),
            "M": z_m.argmax(dim=1),
            "D": z_d.argmax(dim=1),
            "F": z_f.argmax(dim=1),
        }

        # --------------------------------------------------------
        # Gradient validation
        # --------------------------------------------------------
        adv = images.clone().detach().requires_grad_(True)

        normalized_adv = normalize(adv)

        output = multi_lrsf_inv3(
            model,
            normalized_adv,
            num_iters=args.lrs_num_iters,
            compression_rate_shallow=0.8,
            rank_ratio_shallow=0.01,
            compression_rate_balanced=0.5,
            rank_ratio_balanced=0.04,
            compression_rate_deep=0.0,
            rank_ratio_deep=0.1,
            lf_threshold=args.lf_threshold,
            mf_threshold=args.mf_threshold,
        )

        with torch.no_grad():
            original_features = mi.extract_inv3_frequency_features(
                model,
                x.detach(),
            )

        adversarial_features = mi.extract_inv3_frequency_features(
            model,
            normalized_adv,
        )

        original_features = tuple(
            feature.detach()
            for feature in original_features
        )

        freq_loss = mi.frequency_objective(
            adversarial_features,
            original_features,
            schedule="lrsf",
            lf_threshold=args.lf_threshold,
            mf_threshold=args.mf_threshold,
        )

        ce = F.cross_entropy(
            output,
            labels,
        )

        total_loss = ce + freq_loss

        grad = torch.autograd.grad(
            total_loss,
            adv,
            retain_graph=False,
            create_graph=False,
        )[0]

        grad_finite = bool(
            torch.isfinite(grad).all().item()
        )

        grad_mean_abs = grad.abs().mean().item()
        grad_l1 = grad.abs().sum().item()
        grad_l2 = torch.norm(grad).item()

        # --------------------------------------------------------
        # One-step MI-FGSM / epsilon sanity check
        # --------------------------------------------------------
        alpha = args.eps / 10.0

        one_step = (
            adv.detach()
            + alpha * grad.sign()
        )

        delta = torch.clamp(
            one_step - images,
            min=-args.eps,
            max=args.eps,
        )

        one_step = torch.clamp(
            images + delta,
            min=0.0,
            max=1.0,
        )

        one_step_linf = (
            one_step - images
        ).abs().max().item()

        # --------------------------------------------------------
        # Per-expert logit differences
        # --------------------------------------------------------
        expert_diff = {
            "O_S": (z_o - z_s).abs().mean().item(),
            "O_M": (z_o - z_m).abs().mean().item(),
            "O_D": (z_o - z_d).abs().mean().item(),
            "S_M": (z_s - z_m).abs().mean().item(),
            "S_D": (z_s - z_d).abs().mean().item(),
            "M_D": (z_m - z_d).abs().mean().item(),
        }

        row = {
            "batch": batch_idx,
            "image": names[0],
            "shape_shallow": str(tuple(debug["xs"].shape)),
            "shape_middle": str(tuple(debug["xm"].shape)),
            "shape_deep": str(tuple(debug["xd"].shape)),
            "mask_partition_max_error": partition_error,
            "mask_overlap_max": overlap_error,
            "dct_idct_max_error": dct_reconstruction_error,
            "fusion_max_abs_error": fusion_error,
            "shallow_HF_ratio": shallow_h,
            "middle_MF_ratio": middle_m,
            "deep_LF_ratio": deep_l,
            "ce_loss": ce_loss,
            "frequency_loss": freq_loss.item(),
            "total_loss": total_loss.item(),
            "gradient_mean_abs": grad_mean_abs,
            "gradient_L1": grad_l1,
            "gradient_L2": grad_l2,
            "gradient_finite": grad_finite,
            "one_step_Linf": one_step_linf,
            "pred_O": int(predictions["O"][0].item()),
            "pred_S": int(predictions["S"][0].item()),
            "pred_M": int(predictions["M"][0].item()),
            "pred_D": int(predictions["D"][0].item()),
            "pred_F": int(predictions["F"][0].item()),
            "logit_diff_O_S_mean": expert_diff["O_S"],
            "logit_diff_O_M_mean": expert_diff["O_M"],
            "logit_diff_O_D_mean": expert_diff["O_D"],
            "logit_diff_S_M_mean": expert_diff["S_M"],
            "logit_diff_S_D_mean": expert_diff["S_D"],
            "logit_diff_M_D_mean": expert_diff["M_D"],
        }

        rows.append(row)

        print(
            f"[{batch_idx + 1}/{len(loader)}] {names[0]}"
        )
        print(
            f"  shapes: "
            f"S={tuple(debug['xs'].shape)}, "
            f"M={tuple(debug['xm'].shape)}, "
            f"D={tuple(debug['xd'].shape)}"
        )
        print(
            f"  mask partition err : {partition_error:.3e}"
        )
        print(
            f"  mask overlap       : {overlap_error:.3e}"
        )
        print(
            f"  DCT/IDCT error     : {dct_reconstruction_error:.3e}"
        )
        print(
            f"  fusion error       : {fusion_error:.3e}"
        )
        print(
            f"  selected frequency : "
            f"S-H={shallow_h:.6f}, "
            f"M-M={middle_m:.6f}, "
            f"D-L={deep_l:.6f}"
        )
        print(
            f"  losses             : "
            f"CE={ce_loss:.6f}, "
            f"Freq={freq_loss.item():.6f}, "
            f"Total={total_loss.item():.6f}"
        )
        print(
            f"  gradient finite    : {grad_finite}"
        )
        print(
            f"  one-step Linf      : {one_step_linf:.6f} "
            f"(eps={args.eps:.6f})"
        )
        print()

        del adv
        del output
        del adversarial_features
        del grad

        if device == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No rows were produced.")

    os.makedirs(
        os.path.dirname(args.output_csv),
        exist_ok=True,
    )

    with open(
        args.output_csv,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------
    # Aggregate PASS/FAIL
    # ------------------------------------------------------------
    max_partition = max(
        r["mask_partition_max_error"]
        for r in rows
    )
    max_overlap = max(
        r["mask_overlap_max"]
        for r in rows
    )
    max_reconstruction = max(
        r["dct_idct_max_error"]
        for r in rows
    )
    max_fusion = max(
        r["fusion_max_abs_error"]
        for r in rows
    )
    all_grad_finite = all(
        r["gradient_finite"]
        for r in rows
    )
    max_linf = max(
        r["one_step_Linf"]
        for r in rows
    )

    print("=" * 70)
    print("STEP 9A SUMMARY")
    print("=" * 70)

    checks = [
        (
            "LF/MF/HF partition",
            max_partition < 1e-6 and max_overlap < 1e-6,
            f"partition={max_partition:.3e}, "
            f"overlap={max_overlap:.3e}",
        ),
        (
            "DCT -> IDCT reconstruction",
            max_reconstruction < 1e-4,
            f"max_error={max_reconstruction:.3e}",
        ),
        (
            "Manual graph == production graph",
            max_fusion < 1e-4,
            f"max_abs_error={max_fusion:.3e}",
        ),
        (
            "Gradient finite",
            all_grad_finite,
            "all batches finite",
        ),
        (
            "Linf constraint",
            max_linf <= args.eps + 1e-6,
            f"max={max_linf:.6f}, eps={args.eps:.6f}",
        ),
    ]

    passed = 0
    for name, ok, detail in checks:
        print(
            f"[{'PASS' if ok else 'FAIL'}] "
            f"{name}: {detail}"
        )
        if ok:
            passed += 1

    print()
    print(
        f"Mechanistic checks: {passed}/{len(checks)} PASS"
    )
    print("Saved:", args.output_csv)

    if passed == len(checks):
        print("STEP 9A STATUS: PASS")
    else:
        print("STEP 9A STATUS: NEEDS INVESTIGATION")


if __name__ == "__main__":
    main()
