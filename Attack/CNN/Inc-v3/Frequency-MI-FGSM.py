import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

import pretrainedmodels

sys.path.append("../../../")

from function.loader import ImageNet
from function.Normalize import Normalize
from function.dct import dct_2d, idct_2d



# ============================================================
# Frequency configuration
# ============================================================

# Config B selected from Step 4
LF_THRESHOLD = 0.20
MF_THRESHOLD = 0.50


# ============================================================
# Frequency masks
# ============================================================

def radial_frequency_mask(h, w, device):
    """
    Create normalized radial frequency masks.

    Returns:
        lf_mask: low-frequency mask
        mf_mask: mid-frequency mask
        hf_mask: high-frequency mask
    """

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )

    cy = h // 2
    cx = w // 2

    dist = torch.sqrt(
        (yy.float() - cy) ** 2 +
        (xx.float() - cx) ** 2
    )

    max_dist = torch.sqrt(
        torch.tensor(
            float((h // 2) ** 2 + (w // 2) ** 2),
            device=device
        )
    )

    radius = dist / max_dist

    lf_mask = (radius <= LF_THRESHOLD).float()
    mf_mask = (
        (radius > LF_THRESHOLD) &
        (radius <= MF_THRESHOLD)
    ).float()
    hf_mask = (radius > MF_THRESHOLD).float()

    return lf_mask, mf_mask, hf_mask


def apply_frequency_mask(x, mask):
    """
    Apply a frequency mask to a tensor.

    x:
        [B, C, H, W]

    mask:
        [H, W]
    """

    return x * mask.unsqueeze(0).unsqueeze(0)


# ============================================================
# Inception-v3 frequency forward
# ============================================================

class FrequencyInceptionV3(nn.Module):
    """
    Frequency-only surrogate.

    Three depth-aware frequency branches:

        shallow -> High Frequency
        middle  -> Mid Frequency
        deep    -> Low Frequency

    The original LRS decomposition is NOT used here.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def _logit(self, x):
        """
        Inception-v3 classifier head used by LRS-Attack.
        """

        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)

        x = F.avg_pool2d(x, kernel_size=8)
        x = self.model.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.model.last_linear(x)

        return x

    def forward(self, x):
        # ----------------------------------------------------
        # Common stem -> Mixed_5b
        # ----------------------------------------------------

        x = self.model.Conv2d_1a_3x3(x)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = self.model.maxpool1(x)

        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = self.model.maxpool2(x)

        x = self.model.Mixed_5b(x)

        # ====================================================
        # SHALLOW BRANCH
        # Mixed_5b -> DCT -> HF -> IDCT
        # ====================================================

        shallow_dct = dct_2d(x)

        _, _, h_s, w_s = shallow_dct.shape

        _, _, hf_mask_s = radial_frequency_mask(
            h_s,
            w_s,
            shallow_dct.device
        )

        shallow_hf = apply_frequency_mask(
            shallow_dct,
            hf_mask_s
        )

        shallow = idct_2d(shallow_hf)

        shallow = self.model.Mixed_5c(shallow)
        shallow = self.model.Mixed_5d(shallow)

        shallow = self.model.Mixed_6a(shallow)
        shallow = self.model.Mixed_6b(shallow)
        shallow = self.model.Mixed_6c(shallow)
        shallow = self.model.Mixed_6d(shallow)
        shallow = self.model.Mixed_6e(shallow)

        logit_shallow = self._logit(shallow)

        # ====================================================
        # MIDDLE BRANCH
        # Mixed_5b -> Mixed_5c -> Mixed_5d
        # -> DCT -> MF -> IDCT
        # ====================================================

        middle = self.model.Mixed_5c(x)
        middle = self.model.Mixed_5d(middle)

        middle_dct = dct_2d(middle)

        _, _, h_m, w_m = middle_dct.shape

        _, mf_mask_m, _ = radial_frequency_mask(
            h_m,
            w_m,
            middle_dct.device
        )

        middle_mf = apply_frequency_mask(
            middle_dct,
            mf_mask_m
        )

        middle = idct_2d(middle_mf)

        middle = self.model.Mixed_6a(middle)
        middle = self.model.Mixed_6b(middle)
        middle = self.model.Mixed_6c(middle)
        middle = self.model.Mixed_6d(middle)
        middle = self.model.Mixed_6e(middle)

        logit_middle = self._logit(middle)

        # ====================================================
        # DEEP BRANCH
        # Mixed_5b -> Mixed_5c -> Mixed_5d
        # -> Mixed_6a..6e -> DCT -> LF -> IDCT
        # ====================================================

        deep = self.model.Mixed_5c(x)
        deep = self.model.Mixed_5d(deep)

        deep = self.model.Mixed_6a(deep)
        deep = self.model.Mixed_6b(deep)
        deep = self.model.Mixed_6c(deep)
        deep = self.model.Mixed_6d(deep)
        deep = self.model.Mixed_6e(deep)

        deep_dct = dct_2d(deep)

        _, _, h_d, w_d = deep_dct.shape

        lf_mask_d, _, _ = radial_frequency_mask(
            h_d,
            w_d,
            deep_dct.device
        )

        deep_lf = apply_frequency_mask(
            deep_dct,
            lf_mask_d
        )

        deep = idct_2d(deep_lf)

        logit_deep = self._logit(deep)

        # ====================================================
        # Fuse three frequency experts
        # ====================================================

        logits = (
            logit_shallow +
            logit_middle +
            logit_deep
        ) / 3.0

        return logits


# ============================================================
# Model
# ============================================================

def load_model():
    print("Loading pretrainedmodels Inception-v3...")

    model = pretrainedmodels.inceptionv3(
        num_classes=1000,
        pretrained="imagenet"
    )

    model.eval()

    return model.cuda()


# ============================================================
# Attack
# ============================================================

def frequency_mi_fgsm(
    model,
    images,
    labels,
    max_epsilon=16,
    num_iter=10,
    momentum=1.0
):
    """
    Frequency-only MI-FGSM.

    epsilon:
        max_epsilon / 255

    alpha:
        epsilon / num_iter
    """

    epsilon = max_epsilon / 255.0
    alpha = epsilon / num_iter

    adv_images = images.clone().detach()

    momentum_gradient = torch.zeros_like(adv_images)

    for _ in range(num_iter):

        adv_images.requires_grad_(True)

        logits = model(adv_images)

        loss = F.cross_entropy(logits, labels)

        grad = torch.autograd.grad(
            loss,
            adv_images,
            retain_graph=False,
            create_graph=False
        )[0]

        # ----------------------------------------------------
        # MI-FGSM gradient normalization
        # ----------------------------------------------------

        grad_norm = torch.mean(
            torch.abs(grad),
            dim=(1, 2, 3),
            keepdim=True
        )

        grad = grad / (grad_norm + 1e-12)

        momentum_gradient = (
            momentum * momentum_gradient +
            grad
        )

        # ----------------------------------------------------
        # Untargeted attack
        # ----------------------------------------------------

        adv_images = (
            adv_images.detach() +
            alpha * momentum_gradient.sign()
        )

        # ----------------------------------------------------
        # Project into epsilon-ball
        # ----------------------------------------------------

        delta = torch.clamp(
            adv_images - images,
            min=-epsilon,
            max=epsilon
        )

        adv_images = torch.clamp(
            images + delta,
            min=0.0,
            max=1.0
        ).detach()

    return adv_images


# ============================================================
# Image saving
# ============================================================

def save_images(images, image_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    images = images.detach().cpu()

    for image, image_name in zip(images, image_names):

        image = image.permute(1, 2, 0).numpy()

        image = np.clip(
            image * 255.0,
            0,
            255
        ).astype(np.uint8)

        path = os.path.join(
            output_dir,
            image_name
        )

        Image.fromarray(image).save(path)


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Frequency-only MI-FGSM"
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default="../../../dataset/test10.csv"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="../../../dataset/images"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/incv3-Frequency-MI"
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

    parser.add_argument(
        "--max_epsilon",
        type=float,
        default=16
    )

    parser.add_argument(
        "--num_iter",
        type=int,
        default=10
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor()
    ])

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = ImageNet(
        dir=args.input_dir,
        csv_path=args.input_csv,
        transforms=transform
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    print("Images loaded:", len(dataset))

    if len(dataset) == 0:
        raise RuntimeError(
            "No valid images were loaded."
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    base_model = load_model()

    # Freeze model parameters.
    # We still need gradients w.r.t. input images.
    for param in base_model.parameters():
        param.requires_grad = False

    model = FrequencyInceptionV3(base_model)
    model.eval()

    # --------------------------------------------------------
    # Attack
    # --------------------------------------------------------

    os.makedirs(args.output_dir, exist_ok=True)

    for images, image_names, labels in tqdm(
        dataloader,
        desc="Frequency-MI-FGSM"
    ):

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        adv_images = frequency_mi_fgsm(
            model=model,
            images=images,
            labels=labels,
            max_epsilon=args.max_epsilon,
            num_iter=args.num_iter,
            momentum=1.0
        )

        save_images(
            adv_images,
            image_names,
            args.output_dir
        )

    print()
    print("Attack finished.")
    print("Output:", args.output_dir)


if __name__ == "__main__":
    main()