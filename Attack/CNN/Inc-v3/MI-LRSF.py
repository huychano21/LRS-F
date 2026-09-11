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
from function.LRS import multi_lrsf_inv3


# ============================================================
# Dataset
# ============================================================

# class ImageNetDataset(Dataset):

#     def __init__(
#         self,
#         input_dir,
#         input_csv,
#         image_size=299
#     ):

#         self.input_dir = input_dir
#         self.image_size = image_size

#         self.transform = T.Compose([
#             T.Resize(
#                 (
#                     image_size,
#                     image_size
#                 )
#             ),
#             T.ToTensor()
#         ])

#         self.samples = []

#         with open(
#             input_csv,
#             "r"
#         ) as f:

#             reader = csv.reader(f)

#             for row in reader:

#                 if len(row) == 0:
#                     continue

#                 image_id = row[0].strip()

#                 # Skip CSV header
#                 if image_id.lower() in [
#                     "imageid",
#                     "filename",
#                     "image",
#                     "path"
#                 ]:
#                     continue

#                 candidates = [
#                     image_id,
#                     image_id + ".png",
#                     image_id + ".jpg",
#                     image_id + ".jpeg",
#                     image_id + ".JPEG",
#                     image_id + ".PNG"
#                 ]

#                 path = None

#                 for candidate in candidates:

#                     candidate_path = os.path.join(
#                         input_dir,
#                         candidate
#                     )

#                     if os.path.exists(
#                         candidate_path
#                     ):

#                         path = candidate_path
#                         break

#                 if path is None:

#                     print(
#                         "WARNING: missing:",
#                         image_id
#                     )

#                     continue

#                 if len(row) < 2:
#                     raise ValueError(
#                         f"Missing label for image: {image_id}"
#                     )

#                 label = int(
#                     row[1]
#                 )

#                 self.samples.append(
#                     (
#                         path,
#                         image_id,
#                         label
#                     )
#                 )

#     def __len__(self):

#         return len(
#             self.samples
#         )

#     def __getitem__(
#         self,
#         idx
#     ):

#         path, image_id, label = (
#             self.samples[idx]
#         )

#         img = Image.open(
#             path
#         ).convert(
#             "RGB"
#         )

#         img = self.transform(
#             img
#         )

#         return (
#             img,
#             image_id,
#             label
#         )


# ============================================================
# MI-FGSM + LRS-F
# ============================================================

# ============================================================
# Frequency branch
# ============================================================

def build_dct_matrix(n, device, dtype):
    """Orthogonal DCT-II matrix."""
    k = torch.arange(n, device=device, dtype=dtype).view(-1, 1)
    i = torch.arange(n, device=device, dtype=dtype).view(1, -1)

    matrix = torch.cos(
        torch.pi / n * (i + 0.5) * k
    )

    matrix[0] *= 1.0 / np.sqrt(n)
    if n > 1:
        matrix[1:] *= np.sqrt(2.0 / n)

    return matrix


def dct2(x):
    """
    Differentiable 2-D orthogonal DCT-II over the last two dimensions.
    Input:  [B, C, H, W]
    Output: [B, C, H, W]
    """
    h, w = x.shape[-2:]

    dct_h = build_dct_matrix(
        h,
        x.device,
        x.dtype
    )

    dct_w = build_dct_matrix(
        w,
        x.device,
        x.dtype
    )

    y = torch.matmul(
        dct_h,
        x
    )

    y = torch.matmul(
        y,
        dct_w.t()
    )

    return y


def make_frequency_masks(h, w, lf_threshold, mf_threshold, device, dtype):
    """
    Config B:
        LF <= 0.20
        MF > 0.20 and <= 0.50
        HF > 0.50

    The radial frequency is normalized to [0, 1].
    """
    u = torch.arange(
        h,
        device=device,
        dtype=dtype
    ).view(-1, 1)

    v = torch.arange(
        w,
        device=device,
        dtype=dtype
    ).view(1, -1)

    u = u / max(h - 1, 1)
    v = v / max(w - 1, 1)

    radius = torch.sqrt(
        u * u + v * v
    ) / np.sqrt(2.0)

    lf = radius <= lf_threshold
    mf = (radius > lf_threshold) & (radius <= mf_threshold)
    hf = radius > mf_threshold

    return (
        lf.unsqueeze(0).unsqueeze(0).to(dtype),
        mf.unsqueeze(0).unsqueeze(0).to(dtype),
        hf.unsqueeze(0).unsqueeze(0).to(dtype)
    )


def frequency_band_loss(
    feature_delta,
    band_mask
):
    """
    Energy of the perturbation in one DCT frequency band.

    We normalize by the total DCT energy so that feature maps with
    different channel/spatial sizes contribute on a comparable scale.
    """
    coeff = dct2(feature_delta)

    band_energy = torch.sum(
        (coeff * band_mask) ** 2,
        dim=(1, 2, 3)
    )

    total_energy = torch.sum(
        coeff ** 2,
        dim=(1, 2, 3)
    )

    ratio = band_energy / (
        total_energy + 1e-12
    )

    return ratio.mean()


def extract_inv3_frequency_features(
    model,
    normalized_input
):
    """
    Extract the three Inception-v3 feature locations used by Step 3/4:

        shallow  -> Mixed_5b  (35 x 35)
        balanced -> Mixed_5d  (35 x 35)
        deep     -> Mixed_6e  (17 x 17)

    This follows the original Inception-v3 forward path and does not
    apply LRS decomposition. The frequency branch therefore measures
    how the adversarial perturbation changes the original feature
    representation at the selected depths.
    """
    x = model.Conv2d_1a_3x3(normalized_input)
    x = model.Conv2d_2a_3x3(x)
    x = model.Conv2d_2b_3x3(x)
    x = F.max_pool2d(
        x,
        kernel_size=3,
        stride=2
    )
    x = model.Conv2d_3b_1x1(x)
    x = model.Conv2d_4a_3x3(x)
    x = F.max_pool2d(
        x,
        kernel_size=3,
        stride=2
    )

    shallow = model.Mixed_5b(x)

    x = model.Mixed_5c(shallow)
    balanced = model.Mixed_5d(x)

    x = model.Mixed_6a(balanced)
    x = model.Mixed_6b(x)
    x = model.Mixed_6c(x)
    x = model.Mixed_6d(x)
    deep = model.Mixed_6e(x)

    return shallow, balanced, deep


def frequency_objective(
    adv_features,
    orig_features,
    schedule="lrsf",
    lf_threshold=0.20,
    mf_threshold=0.50
):
    """
    Step-8 frequency ablation schedules.

    The table is encoded as: 
        LRS-F          : shallow H, middle M, deep L
        LRS + H only   : shallow H, middle none, deep none
        LRS + M only   : shallow none, middle M, deep none
        LRS + L only   : shallow none, middle none, deep L
        Wrong schedule : shallow L, middle M, deep H

    Each selected band contributes the normalized DCT energy ratio of
    the adversarial feature perturbation at that depth. Unselected
    depths contribute zero.
    """
    adv_shallow, adv_balanced, adv_deep = adv_features
    orig_shallow, orig_balanced, orig_deep = orig_features

    masks_shallow = make_frequency_masks(
        adv_shallow.shape[-2], adv_shallow.shape[-1],
        lf_threshold, mf_threshold, adv_shallow.device, adv_shallow.dtype
    )
    masks_balanced = make_frequency_masks(
        adv_balanced.shape[-2], adv_balanced.shape[-1],
        lf_threshold, mf_threshold, adv_balanced.device, adv_balanced.dtype
    )
    masks_deep = make_frequency_masks(
        adv_deep.shape[-2], adv_deep.shape[-1],
        lf_threshold, mf_threshold, adv_deep.device, adv_deep.dtype
    )

    delta_shallow = adv_shallow - orig_shallow
    delta_balanced = adv_balanced - orig_balanced
    delta_deep = adv_deep - orig_deep

    schedule = schedule.lower().strip()
    schedule_map = {
        "lrsf": ("h", "m", "l"),
        "lrs+h": ("h", None, None),
        "lrs_h": ("h", None, None),
        "h": ("h", None, None),
        "lrs+m": (None, "m", None),
        "lrs_m": (None, "m", None),
        "m": (None, "m", None),
        "lrs+l": (None, None, "l"),
        "lrs_l": (None, None, "l"),
        "l": (None, None, "l"),
        "wrong": ("l", "m", "h"),
        "wrong_schedule": ("l", "m", "h"),
    }

    if schedule not in schedule_map:
        raise ValueError(
            f"Unknown frequency schedule: {schedule}. "
            "Use lrsf, lrs_h, lrs_m, lrs_l, or wrong_schedule."
        )

    selected = schedule_map[schedule]
    mask_lookup = {
        "l": 0,
        "m": 1,
        "h": 2,
    }

    losses = []
    active = 0

    deltas = [
        delta_shallow,
        delta_balanced,
        delta_deep
    ]
    masks = [
        masks_shallow,
        masks_balanced,
        masks_deep
    ]

    for delta, depth_masks, band in zip(deltas, masks, selected):
        if band is None:
            continue
        losses.append(
            frequency_band_loss(
                delta,
                depth_masks[mask_lookup[band]]
            )
        )
        active += 1

    if active == 0:
        return adv_shallow.sum() * 0.0

    return sum(losses) / float(active)


# ============================================================
# MI-FGSM + LRS-F + Frequency
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
    frequency_weight=1.0,
    frequency_schedule="lrsf"
):

    alpha = eps / num_iter

    adv = x.clone().detach()

    momentum = torch.zeros_like(
        adv
    )

    # --------------------------------------------------------
    # Original feature representation.
    #
    # These features are detached because the frequency branch
    # measures the change caused by the adversarial image.
    # --------------------------------------------------------

    with torch.no_grad():
        normalized_orig = normalize(x)

        orig_features = extract_inv3_frequency_features(
            model,
            normalized_orig
        )

        orig_features = tuple(
            feat.detach()
            for feat in orig_features
        )

    for _ in range(num_iter):

        adv.requires_grad_(True)

        normalized_adv = normalize(adv)

        # ----------------------------------------------------
        # LRS-F surrogate
        # ----------------------------------------------------

        logits = multi_lrsf_inv3(
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

            lf_threshold=
                lf_threshold,

            mf_threshold=
                mf_threshold
        )

        ce_loss = F.cross_entropy(
            logits,
            labels
        )

        # ----------------------------------------------------
        # Frequency branch
        #
        # Extract original-path Inception features and measure
        # normalized DCT energy of the feature perturbation:
        #
        #   shallow  -> HF
        #   balanced -> MF
        #   deep     -> LF
        #
        # Maximizing this term encourages the adversarial feature
        # change to occupy the selected frequency bands.
        # ----------------------------------------------------

        adv_features = extract_inv3_frequency_features(
            model,
            normalized_adv
        )

        freq_loss = frequency_objective(
            adv_features,
            orig_features,
            schedule=frequency_schedule,
            lf_threshold=lf_threshold,
            mf_threshold=mf_threshold
        )

        # Both CE and frequency objective are maximized by the
        # untargeted MI-FGSM ascent step.
        loss = (
            ce_loss
            + frequency_weight * freq_loss
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

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    parser.add_argument(

        "--input_csv",

        type=str,

        default=os.path.join(
            PROJECT_ROOT,
            "dataset/test10.csv"
        )
    )

    parser.add_argument(

        "--input_dir",

        type=str,

        default=os.path.join(
            PROJECT_ROOT,
            "dataset/images"
        )
    )

    parser.add_argument(

        "--output_dir",

        type=str,

        default=os.path.join(
            PROJECT_ROOT,
            "Attack/outputs/incv3-Step8-ablation"
        )
    )

    parser.add_argument(

        "--batch_size",

        type=int,

        default=1
    )

    parser.add_argument(

        "--num_workers",

        type=int,

        default=0
    )

    # --------------------------------------------------------
    # MI-FGSM
    # --------------------------------------------------------

    parser.add_argument(

        "--max_epsilon",

        type=float,

        default=16.0
    )

    parser.add_argument(

        "--num_iter",

        type=int,

        default=10
    )

    parser.add_argument(

        "--momentum",

        type=float,

        default=1.0
    )

    # --------------------------------------------------------
    # LRS decomposition
    # --------------------------------------------------------

    parser.add_argument(

        "--lrs_num_iters",

        type=int,

        default=5
    )

    parser.add_argument(

        "--compression_rate_shallow",

        type=float,

        default=0.8
    )

    parser.add_argument(

        "--rank_ratio_shallow",

        type=float,

        default=0.01
    )

    parser.add_argument(

        "--compression_rate_balanced",

        type=float,

        default=0.5
    )

    parser.add_argument(

        "--rank_ratio_balanced",

        type=float,

        default=0.04
    )

    parser.add_argument(

        "--compression_rate_deep",

        type=float,

        default=0.0
    )

    parser.add_argument(

        "--rank_ratio_deep",

        type=float,

        default=0.1
    )

    # --------------------------------------------------------
    # Frequency partition
    # --------------------------------------------------------

    parser.add_argument(

        "--lf_threshold",

        type=float,

        default=0.20
    )

    parser.add_argument(

        "--mf_threshold",

        type=float,

        default=0.50
    )

    parser.add_argument(

        "--frequency_weight",

        type=float,

        default=1.0
    )

    parser.add_argument(

        "--frequency_schedule",

        type=str,

        default="lrsf",

        choices=[
            "lrsf",
            "lrs_h",
            "lrs_m",
            "lrs_l",
            "wrong_schedule"
        ]
    )

    parser.add_argument(

        "--optimizer",

        type=str,

        default="mi",

        choices=["mi", "ifgsm"]
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    parser.add_argument(

        "--seed",

        type=int,

        default=123
    )

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
        f"Frequency objective weight: {opt.frequency_weight}"
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

            frequency_weight=
                opt.frequency_weight,

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