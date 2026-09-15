# NOTE:
# This driver matches the proposed feature-level LRS-F flow.
# It requires function/LRS.py to provide:
#     multi_lrsf_featurefusion_inv3(...)
# which must perform low-rank/sparse decomposition, DCT band extraction,
# depth-aware feature fusion, inverse rescaling, expert forwarding, and
# logit fusion. The original MI-LRSF.py only supplied an auxiliary
# frequency loss and therefore did not implement the proposed flow.

import os
import sys
# import csv
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import pretrainedmodels

# ============================================================
# Add project root to Python path
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        "../../../"
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# Project imports
# ============================================================

from function.loader import ImageNet
from function.Normalize import Normalize
from function.LRS import multi_lrsf_featurefusion_inv3

   

# ============================================================
# MI-FGSM + Proposed LRS-F (feature-level frequency fusion)
# ============================================================

def attack(
    model,
    normalize,
    x,
    labels,
    eps=16.0 / 255.0,
    num_iter=10,
    momentum_decay=1.0,
    lrs_num_iters=5,
    compression_rate_shallow=0.8,
    rank_ratio_shallow=0.01,
    compression_rate_balanced=0.5,
    rank_ratio_balanced=0.04,
    compression_rate_deep=0.0,
    rank_ratio_deep=0.1,
    lf_threshold=0.20,
    mf_threshold=0.50,
    gamma_shallow=1.0,
    gamma_middle=1.0,
    gamma_deep=1.0,
    frequency_schedule="lrsf"
):

    alpha = eps / num_iter

    adv = x.clone().detach()

    momentum = torch.zeros_like(
        adv
    )

    # Frequency is fused inside the hierarchical experts; no clean-feature
    # reference branch is needed in this driver.

    for _ in range(num_iter):

        adv.requires_grad_(True)

        normalized_adv = normalize(adv)

        # ----------------------------------------------------
        # Proposed LRS-F: feature-level frequency fusion
        #
        # Expected behavior inside multi_lrsf_featurefusion_inv3:
        #   shallow: R_s = S_s + gamma_shallow * F_s^high
        #   middle : R_m = L_m + S_m + gamma_middle * F_m^mid
        #   deep   : R_d = L_d + gamma_deep * F_d^low
        #
        # Then inverse-rescale each R_l, forward the remaining
        # layers, fuse original/shallow/middle/deep logits, and
        # return the fused logits. No auxiliary frequency loss is
        # used in this driver; frequency affects the attack through
        # the expert representations themselves.
        # ----------------------------------------------------

        logits = multi_lrsf_featurefusion_inv3(
            model,
            normalized_adv,

            num_iters=lrs_num_iters,

            compression_rate_shallow=
                compression_rate_shallow,

            rank_ratio_shallow=
                rank_ratio_shallow,

            compression_rate_balanced=
                compression_rate_balanced,

            rank_ratio_balanced=
                rank_ratio_balanced,

            compression_rate_deep=
                compression_rate_deep,

            rank_ratio_deep=
                rank_ratio_deep,

            lf_threshold=lf_threshold,
            mf_threshold=mf_threshold,

            gamma_shallow=gamma_shallow,
            gamma_middle=gamma_middle,
            gamma_deep=gamma_deep,

            frequency_schedule=frequency_schedule
        )

        loss = F.cross_entropy(
            logits,
            labels
        )

        grad = torch.autograd.grad(
            loss,
            adv,
            retain_graph=False,
            create_graph=False
        )[0]

        # ----------------------------------------------------
        # MI-FGSM gradient normalization
        # ----------------------------------------------------

        grad = grad / (
            torch.mean(
                torch.abs(grad),
                dim=(1, 2, 3),
                keepdim=True
            )
            + 1e-12
        )

        # ----------------------------------------------------
        # Momentum
        # ----------------------------------------------------

        momentum = (
            momentum_decay * momentum
            + grad
        )

        # ----------------------------------------------------
        # MI-FGSM update
        # ----------------------------------------------------

        adv = (
            adv.detach()
            + alpha * momentum.sign()
        )

        # ----------------------------------------------------
        # epsilon constraint
        # ----------------------------------------------------

        delta = torch.clamp(
            adv - x,
            min=-eps,
            max=eps
        )

        adv = torch.clamp(
            x + delta,
            min=0,
            max=1
        ).detach()

    return adv


# ============================================================
# Save adversarial images
# ============================================================

def save_images(
    images,
    names,
    output_dir
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    images = (

        images
        .detach()
        .cpu()
        .numpy()

        .transpose(
            0,
            2,
            3,
            1
        )

        * 255.0
    )

    images = np.clip(

        images,

        0,
        255

    ).astype(
        np.uint8
    )

    for img, name in zip(
        images,
        names
    ):

        path = os.path.join(
            output_dir,
            name
        )

        Image.fromarray(
            img
        ).save(
            path
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default=os.path.join(PROJECT_ROOT,"dataset/test10.csv"))
    parser.add_argument("--input_dir", type=str, default=os.path.join(PROJECT_ROOT,"dataset/images"))
    parser.add_argument("--output_dir",type=str, default=os.path.join(PROJECT_ROOT,"Attack/outputs/incv3-Step8-ablation"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

   
    # MI-FGSM
    parser.add_argument("--max_epsilon", type=float, default=16.0)
    parser.add_argument("--num_iter", type=int, default=10)
    parser.add_argument( "--momentum", type=float, default=1.0)

    # LRS decomposition
    parser.add_argument( "--lrs_num_iters", type=int, default=5)
    parser.add_argument( "--compression_rate_shallow", type=float, default=0.8)
    parser.add_argument( "--rank_ratio_shallow", type=float, default=0.01)
    parser.add_argument( "--compression_rate_balanced", type=float, default=0.5)
    parser.add_argument( "--rank_ratio_balanced", type=float, default=0.04)
    parser.add_argument( "--compression_rate_deep", type=float, default=0.0)
    parser.add_argument( "--rank_ratio_deep", type=float, default=0.1)

    # --------------------------------------------------------
    # Frequency partition
    # --------------------------------------------------------

    parser.add_argument( "--lf_threshold", type=float, default=0.20)
    parser.add_argument( "--mf_threshold", type=float, default=0.50)
    parser.add_argument( "--gamma_shallow", type=float, default=1.0)
    parser.add_argument( "--gamma_middle", type=float, default=1.0)
    parser.add_argument( "--gamma_deep", type=float, default=1.0)
    parser.add_argument( "--frequency_schedule", type=str, default="lrsf", choices=[
            "lrsf",
            "lrs_h",
            "lrs_m",
            "lrs_l",
            "wrong_schedule"
        ])
    parser.add_argument( "--optimizer", type=str, default="mi", choices=["mi", "ifgsm"])

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    parser.add_argument( "--seed", type=int, default=123)
    opt = parser.parse_args()

    # ========================================================
    # Reproducibility
    # ========================================================

    torch.manual_seed(
        opt.seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            opt.seed
        )

    np.random.seed(
        opt.seed
    )

    random.seed(
        opt.seed
    )

    os.environ[
        "PYTHONHASHSEED"
    ] = str(
        opt.seed
    )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ========================================================
    # Device
    # ========================================================

    device = (

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )

    print(
        "Device:",
        device
    )

    print(
        "Repository:",
        PROJECT_ROOT
    )

    print(
        "Input CSV:",
        opt.input_csv
    )

    print(
        "Input images:",
        opt.input_dir
    )

    print(
        "Output:",
        opt.output_dir
    )

    print(
        "LRS-Frequency config:"
    )

    print(
        f"LF <= {opt.lf_threshold:.2f}, "
        f"MF <= {opt.mf_threshold:.2f}, "
        f"HF > {opt.mf_threshold:.2f}"
    )

    print(
        f"Feature-frequency gammas: shallow={opt.gamma_shallow}, "
        f"middle={opt.gamma_middle}, deep={opt.gamma_deep}"
    )

    print(
        f"Frequency schedule: {opt.frequency_schedule}"
    )

    print(
        f"Optimizer: {opt.optimizer}"
    )

    print(
        "LRS config:"
    )

    print(
        f"shallow  compression="
        f"{opt.compression_rate_shallow}, "
        f"rank={opt.rank_ratio_shallow}"
    )

    print(
        f"balanced compression="
        f"{opt.compression_rate_balanced}, "
        f"rank={opt.rank_ratio_balanced}"
    )

    print(
        f"deep     compression="
        f"{opt.compression_rate_deep}, "
        f"rank={opt.rank_ratio_deep}"
    )

    # ========================================================
    # Dataset
    # ========================================================

    transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor()
    ])

    dataset = ImageNet(
        opt.input_dir,
        opt.input_csv,
        transforms=transform
    )

    print(
        "Images loaded:",
        len(dataset)
    )

    if len(dataset) == 0:

        raise RuntimeError(

            "No valid images were loaded. "

            "Check --input_csv and --input_dir."
        )

    loader = DataLoader(

        dataset,

        batch_size=opt.batch_size,

        shuffle=False,

        num_workers=opt.num_workers,

        pin_memory=(
            device == "cuda"
        )
    )

    # ========================================================
    # Normalization
    #
    # Same preprocessing used by original LRS attack:
    # [0, 1] -> normalized for pretrainedmodels Inception-v3
    # ========================================================

    normalize = Normalize(

        np.array([
            0.5,
            0.5,
            0.5
        ]),

        np.array([
            0.5,
            0.5,
            0.5
        ])

    ).to(
        device
    )

    # ========================================================
    # Surrogate model
    # ========================================================

    print(
        "Loading pretrainedmodels Inception-v3..."
    )

    model = pretrainedmodels.inceptionv3(

        num_classes=1000,

        pretrained="imagenet"
    )

    model = model.to(
        device
    )

    model.eval()

    # Freeze model parameters.
    #
    # We need gradient w.r.t. input image,
    # not gradient w.r.t. model weights.

    for param in model.parameters():

        param.requires_grad_(
            False
        )

    # ========================================================
    # Attack
    # ========================================================

    progress = tqdm(

        loader,

        desc=f"Step8-{opt.frequency_schedule}"
    )

    for images, names, labels in progress:

        images = images.to(

            device,

            non_blocking=True
        )

        labels = labels.to(

            device,

            non_blocking=True
        )

        adv = attack(

            model=model,

            normalize=normalize,

            x=images,

            labels=labels,

            eps=(
                opt.max_epsilon
                / 255.0
            ),

            num_iter=
                opt.num_iter,

            momentum_decay=(
                opt.momentum
                if opt.optimizer == "mi"
                else 0.0
            ),

            lrs_num_iters=
                opt.lrs_num_iters,

            compression_rate_shallow=
                opt.compression_rate_shallow,

            rank_ratio_shallow=
                opt.rank_ratio_shallow,

            compression_rate_balanced=
                opt.compression_rate_balanced,

            rank_ratio_balanced=
                opt.rank_ratio_balanced,

            compression_rate_deep=
                opt.compression_rate_deep,

            rank_ratio_deep=
                opt.rank_ratio_deep,

            lf_threshold=
                opt.lf_threshold,

            mf_threshold=
                opt.mf_threshold,

            gamma_shallow=
                opt.gamma_shallow,

            gamma_middle=
                opt.gamma_middle,

            gamma_deep=
                opt.gamma_deep,

            frequency_schedule=
                opt.frequency_schedule
        )

        save_images(

            adv,

            names,

            opt.output_dir
        )

        # Explicit cleanup for 6 GB GPU

        del adv
        del images
        del labels

        if device == "cuda":

            torch.cuda.empty_cache()

    print()

    print(
        "Attack finished."
    )

    print(
        f"Method: LRS + {opt.frequency_schedule}"
        if opt.frequency_schedule != "lrsf"
        else "Method: MI + LRS-F"
    )

    print(
        "Output:",
        opt.output_dir
    )


if __name__ == "__main__":

    main()