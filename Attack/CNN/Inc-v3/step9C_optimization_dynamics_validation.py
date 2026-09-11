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

# Cấu hình đường dẫn
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from function.loader import ImageNet
from function.Normalize import Normalize
from function.LRS import multi_lrsf_inv3

# Tải module MI-LRSF để sử dụng lại các hàm frequency
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(HERE, "MI-LRSF.py"),
    os.path.join(HERE, "MI-LRSF(2).py"),
]
MI_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
if MI_PATH is None:
    raise FileNotFoundError("Không tìm thấy MI-LRSF.py hoặc MI-LRSF(2).py tại " + HERE)

spec = importlib.util.spec_from_file_location("mi_lrsf", MI_PATH)
mi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mi)

def main():
    parser = argparse.ArgumentParser(description="Step 9C - Optimization Dynamics")
    parser.add_argument("--input_csv", default=os.path.join(PROJECT_ROOT, "dataset/test10.csv"))
    parser.add_argument("--input_dir", default=os.path.join(PROJECT_ROOT, "dataset/images"))
    parser.add_argument("--output_csv", default=os.path.join(PROJECT_ROOT, "Attack/outputs/step9C_optimization_dynamics_validation.csv"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--max_epsilon", type=float, default=16.0)
    parser.add_argument("--num_iter", type=int, default=10)
    parser.add_argument("--lrs_num_iters", type=int, default=5)
    parser.add_argument("--lf_threshold", type=float, default=0.20)
    parser.add_argument("--mf_threshold", type=float, default=0.50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eps = args.max_epsilon / 255.0
    alpha = eps / args.num_iter

    # Cố định random seed
    torch.manual_seed(12345)
    np.random.seed(12345)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(12345)

    transform = T.Compose([T.Resize((299, 299)), T.ToTensor()])
    dataset = ImageNet(args.input_dir, args.input_csv, transforms=transform)
    n = min(args.num_images, len(dataset))
    dataset = torch.utils.data.Subset(dataset, list(range(n)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    normalize = Normalize(np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])).to(device)

    print("=" * 70)
    print("STEP 9C - OPTIMIZATION DYNAMICS VALIDATION")
    print("=" * 70)

    model = pretrainedmodels.inceptionv3(num_classes=1000, pretrained="imagenet").to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    all_rows = []
    eps_respected = True
    grad_finite = True
    loss_finite = True

    for batch_idx, (images, names, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        print(f"Tracking Image [{batch_idx+1}/{len(loader)}]: {names[0]} (True Label: {labels[0].item()})")

        adv = images.clone().detach()
        momentum = torch.zeros_like(adv)

        with torch.no_grad():
            normalized_orig = normalize(images)
            orig_features = mi.extract_inv3_frequency_features(model, normalized_orig)
            orig_features = tuple(f.detach() for f in orig_features)

        for i in range(args.num_iter):
            adv.requires_grad_(True)
            normalized_adv = normalize(adv)

            # Đặt lại seed theo iteration để ổn định hàm altern_ls
            torch.manual_seed(12345 + batch_idx * 100 + i)

            logits = multi_lrsf_inv3(
                model, normalized_adv,
                num_iters=args.lrs_num_iters,
                lf_threshold=args.lf_threshold,
                mf_threshold=args.mf_threshold
            )

            ce_loss = F.cross_entropy(logits, labels)

            adv_features = mi.extract_inv3_frequency_features(model, normalized_adv)
            freq_loss = mi.frequency_objective(
                adv_features, orig_features,
                schedule="lrsf",
                lf_threshold=args.lf_threshold,
                mf_threshold=args.mf_threshold
            )

            total_loss = ce_loss + freq_loss
            if not torch.isfinite(total_loss):
                loss_finite = False

            grad = torch.autograd.grad(total_loss, adv, retain_graph=False, create_graph=False)[0]

            if not torch.isfinite(grad).all():
                grad_finite = False

            grad_l2 = torch.norm(grad.reshape(grad.shape[0], -1), dim=1).mean().item()
            grad_linf = grad.reshape(grad.shape[0], -1).abs().amax(dim=1).mean().item()

            # Normalization và Momentum
            grad_normed = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-12)
            momentum = momentum + grad_normed
            mom_l2 = torch.norm(momentum.reshape(momentum.shape[0], -1), dim=1).mean().item()

            # Cập nhật nhiễu
            adv_next = adv.detach() + alpha * momentum.sign()
            delta = torch.clamp(adv_next - images, min=-eps, max=eps)
            adv_next = torch.clamp(images + delta, min=0.0, max=1.0).detach()

            pert_linf = (adv_next - images).reshape(images.shape[0], -1).abs().amax(dim=1).mean().item()
            if pert_linf > eps + 1e-5:
                eps_respected = False

            pred = logits.argmax(dim=1)[0].item()

            row = {
                "image": names[0],
                "iteration": i + 1,
                "ce_loss": ce_loss.item(),
                "freq_loss": freq_loss.item(),
                "total_loss": total_loss.item(),
                "grad_L2": grad_l2,
                "grad_Linf": grad_linf,
                "momentum_L2": mom_l2,
                "perturbation_Linf": pert_linf,
                "prediction": pred,
                "true_label": labels[0].item()
            }
            all_rows.append(row)
            adv = adv_next

            print(f"  Iter {i+1:2d} | CE: {ce_loss.item():.4f} | Freq: {freq_loss.item():.4f} | "
                  f"Total: {total_loss.item():.4f} | Grad L2: {grad_l2:.4f} | "
                  f"Mom L2: {mom_l2:.4f} | Pert Linf: {pert_linf:.5f} | Pred: {pred}")
        print()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print("=" * 70)
    print("STEP 9C SUMMARY")
    print("=" * 70)
    print(f"[{'PASS' if loss_finite else 'FAIL'}] Losses finite across all iterations")
    print(f"[{'PASS' if grad_finite else 'FAIL'}] Gradients finite across all iterations")
    print(f"[{'PASS' if eps_respected else 'FAIL'}] Epsilon bound strictly respected")
    print("Saved:", args.output_csv)
    print("STEP 9C STATUS:", "PASS" if (loss_finite and grad_finite and eps_respected) else "FAIL")

if __name__ == "__main__":
    main()