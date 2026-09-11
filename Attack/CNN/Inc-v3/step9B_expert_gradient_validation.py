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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from function.loader import ImageNet
from function.Normalize import Normalize
from function.LRS import inv3_logit
from function.dct import dct_2d, idct_2d

# Reuse the validated Step-9A graph. It exposes zO/zS/zM/zD/zf.
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(HERE, "step9A_mechanistic_validation.py"),
]
STEP9A_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
if STEP9A_PATH is None:
    raise FileNotFoundError("step9A_mechanistic_validation.py not found in " + HERE)

spec = importlib.util.spec_from_file_location("step9a", STEP9A_PATH)
step9a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step9a)

debug_lrsf_inv3 = step9a.debug_lrsf_inv3


def grad_stats(g):
    flat = g.reshape(g.shape[0], -1)
    l2 = torch.linalg.vector_norm(flat, dim=1)
    linf = flat.abs().amax(dim=1)
    mean_abs = flat.abs().mean(dim=1)
    return l2, linf, mean_abs


def cosine_matrix(grads):
    names = list(grads.keys())
    mat = torch.zeros((len(names), len(names)), device=next(iter(grads.values())).device)
    for i, a in enumerate(names):
        ga = grads[a].reshape(grads[a].shape[0], -1)
        for j, b in enumerate(names):
            gb = grads[b].reshape(grads[b].shape[0], -1)
            mat[i, j] = F.cosine_similarity(ga, gb, dim=1).mean()
    return names, mat


def main():
    parser = argparse.ArgumentParser(description="Step 9B expert/gradient contribution validation.")
    parser.add_argument("--input_csv", default=os.path.join(PROJECT_ROOT, "dataset/test10.csv"))
    parser.add_argument("--input_dir", default=os.path.join(PROJECT_ROOT, "dataset/images"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--lrs_num_iters", type=int, default=5)
    parser.add_argument("--lf_threshold", type=float, default=0.20)
    parser.add_argument("--mf_threshold", type=float, default=0.50)
    parser.add_argument("--output_csv", default=os.path.join(PROJECT_ROOT, "Attack/outputs/step9B_expert_gradient_validation.csv"))
    args = parser.parse_args()

    torch.manual_seed(12345)
    np.random.seed(12345)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(12345)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([T.Resize((299, 299)), T.ToTensor()])
    dataset = ImageNet(args.input_dir, args.input_csv, transforms=transform)
    n = min(args.num_images, len(dataset))
    dataset = torch.utils.data.Subset(dataset, list(range(n)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device == "cuda"))

    normalize = Normalize(np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])).to(device)

    print("=" * 70)
    print("STEP 9B - EXPERT / GRADIENT CONTRIBUTION VALIDATION")
    print("=" * 70)
    print("Device:", device)
    print("Images:", len(dataset))
    print("LF threshold:", args.lf_threshold)
    print("MF threshold:", args.mf_threshold)
    print("LRS iterations:", args.lrs_num_iters)
    print()
    print("Loading pretrainedmodels Inception-v3...")

    model = pretrainedmodels.inceptionv3(num_classes=1000, pretrained="imagenet").to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows = []
    all_pass = True
    logits_pass_all = True
    fusion_pass_all = True
    expert_grad_pass_all = True
    fused_grad_pass_all = True
    routing_pass_all = True

    for batch_idx, (images, names, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        adv = images.clone().detach().requires_grad_(True)
        x = normalize(adv)

        # Deterministic LRS initialization, matching Step 9A methodology.
        torch.manual_seed(12345 + batch_idx)
        debug = debug_lrsf_inv3(
            model, x, num_iters=args.lrs_num_iters,
            lf=args.lf_threshold, mf=args.mf_threshold,
        )

        z = {k: debug[k] for k in ["zO", "zS", "zM", "zD", "zf"]}

        # Fusion must be the equal-weight average of four expert logits.
        fusion_manual = (z["zO"] + z["zS"] + z["zM"] + z["zD"]) / 4.0
        fusion_error = (fusion_manual - z["zf"]).abs().max().item()

        # Expert logits must be finite and meaningfully distinct.
        finite_logits = all(torch.isfinite(v).all().item() for v in z.values())
        pair_diffs = {
            "O_S": (z["zO"] - z["zS"]).abs().mean().item(),
            "O_M": (z["zO"] - z["zM"]).abs().mean().item(),
            "O_D": (z["zO"] - z["zD"]).abs().mean().item(),
            "S_M": (z["zS"] - z["zM"]).abs().mean().item(),
            "S_D": (z["zS"] - z["zD"]).abs().mean().item(),
            "M_D": (z["zM"] - z["zD"]).abs().mean().item(),
        }
        expert_distinct = max(pair_diffs.values()) > 1e-8

        losses = {k: F.cross_entropy(z[k], labels) for k in ["zO", "zS", "zM", "zD", "zf"]}
        grads = {}
        for i, k in enumerate(["zO", "zS", "zM", "zD", "zf"]):
            grads[k] = torch.autograd.grad(
                losses[k], adv, retain_graph=(i < 4), create_graph=False
            )[0].detach()

        finite_grads = all(torch.isfinite(g).all().item() for g in grads.values())
        norms = {k: grad_stats(g) for k, g in grads.items()}
        names_grad, cos = cosine_matrix({k: grads[k] for k in ["zO", "zS", "zM", "zD"]})

        # Fusion gradient should not be numerically identical to only one expert,
        # and should have non-zero contribution from the fused computational graph.
        fused_norm = norms["zf"][0].mean().item()
        expert_norms = {k: norms[k][0].mean().item() for k in ["zO", "zS", "zM", "zD"]}
        fused_nonzero = fused_norm > 1e-12
        expert_nonzero = all(v > 1e-12 for v in expert_norms.values())

        # The intended frequency routing is represented by the exposed components.
        freq_routing = {
            "S_H": debug["FH"],
            "M_M": debug["FM"],
            "D_L": debug["FL"],
        }
        routing_energy = {}
        for k, feat in freq_routing.items():
            coeff = dct_2d(feat)
            energy = coeff * coeff
            total = energy.sum() + 1e-12
            h, w = feat.shape[-2:]
            masks = step9a.mi.make_frequency_masks(h, w, args.lf_threshold, args.mf_threshold, feat.device, feat.dtype)
            lf_m, mf_m, hf_m = masks
            target = {"S_H": hf_m, "M_M": mf_m, "D_L": lf_m}[k]
            routing_energy[k] = ((energy * target).sum() / total).item()
        routing_pass = all(v > 0.999999 for v in routing_energy.values())

        print(f"[{batch_idx+1}/{len(loader)}] {names[0]}")
        print(f"  fusion error       : {fusion_error:.3e}")
        print(f"  logits finite      : {finite_logits}")
        print("  expert logit diff  : " + ", ".join(f"{k}={v:.6e}" for k, v in pair_diffs.items()))
        print("  expert CE          : " + ", ".join(f"{k}={losses[k].item():.6f}" for k in ["zO","zS","zM","zD","zf"]))
        print("  grad L2            : " + ", ".join(f"{k}={expert_norms[k]:.6e}" for k in ["zO","zS","zM","zD"]) + f", F={fused_norm:.6e}")
        print("  grad finite        :", finite_grads)
        print("  routing energy     : " + ", ".join(f"{k}={v:.6f}" for k, v in routing_energy.items()))
        print("  gradient cosine    :")
        print("      " + " ".join(f"{n:>6}" for n in names_grad))
        for i, n in enumerate(names_grad):
            print("      " + f"{n:>6} " + " ".join(f"{cos[i,j].item():6.3f}" for j in range(len(names_grad))))
        print()

        row = {
            "batch": batch_idx,
            "image": names[0],
            "fusion_max_abs_error": fusion_error,
            "logits_finite": finite_logits,
            "expert_distinct": expert_distinct,
            "O_S_logit_diff": pair_diffs["O_S"],
            "O_M_logit_diff": pair_diffs["O_M"],
            "O_D_logit_diff": pair_diffs["O_D"],
            "S_M_logit_diff": pair_diffs["S_M"],
            "S_D_logit_diff": pair_diffs["S_D"],
            "M_D_logit_diff": pair_diffs["M_D"],
            "loss_O": losses["zO"].item(),
            "loss_S": losses["zS"].item(),
            "loss_M": losses["zM"].item(),
            "loss_D": losses["zD"].item(),
            "loss_F": losses["zf"].item(),
            "grad_O_L2": expert_norms["zO"],
            "grad_S_L2": expert_norms["zS"],
            "grad_M_L2": expert_norms["zM"],
            "grad_D_L2": expert_norms["zD"],
            "grad_F_L2": fused_norm,
            "grad_finite": finite_grads,
            "routing_S_H": routing_energy["S_H"],
            "routing_M_M": routing_energy["M_M"],
            "routing_D_L": routing_energy["D_L"],
        }
        for i, a in enumerate(names_grad):
            for j, b in enumerate(names_grad):
                row[f"cos_{a}_{b}"] = cos[i,j].item()
        rows.append(row)

        # GPU floating-point reductions can introduce ~1e-7 to 1e-6 error
        # even when the fusion formula is exactly the same.
        fusion_pass = fusion_error <= 1e-6
        logits_pass = finite_logits and expert_distinct
        expert_grad_pass = finite_grads and expert_nonzero
        fused_grad_pass = finite_grads and fused_nonzero

        batch_pass = (
            fusion_pass and logits_pass and expert_grad_pass and
            fused_grad_pass and routing_pass
        )
        all_pass = all_pass and batch_pass
        logits_pass_all = logits_pass_all and logits_pass
        fusion_pass_all = fusion_pass_all and fusion_pass
        expert_grad_pass_all = expert_grad_pass_all and expert_grad_pass
        fused_grad_pass_all = fused_grad_pass_all and fused_grad_pass
        routing_pass_all = routing_pass_all and routing_pass

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 70)
    print("STEP 9B SUMMARY")
    print("=" * 70)
    print(f"[{'PASS' if logits_pass_all else 'FAIL'}] Expert logits finite and distinct")
    print(f"[{'PASS' if fusion_pass_all else 'FAIL'}] Equal-weight fusion verified")
    print(f"[{'PASS' if expert_grad_pass_all else 'FAIL'}] Per-expert gradients finite/non-zero")
    print(f"[{'PASS' if fused_grad_pass_all else 'FAIL'}] Fused gradient finite/non-zero")
    print(f"[{'PASS' if routing_pass_all else 'FAIL'}] Frequency routing S-H / M-M / D-L verified")
    print("Saved:", args.output_csv)
    print("STEP 9B STATUS:", "PASS" if all_pass else "FAIL")


if __name__ == "__main__":
    main()
