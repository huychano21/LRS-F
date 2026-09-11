import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
import pretrainedmodels
from PIL import Image
from torchvision import transforms
from function.dct import dct_2d


# ============================================================
# Config
# ============================================================

INPUT_CSV = "dataset/images.csv"
INPUT_DIR = "dataset/images"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIGS = {
    "A": (0.15, 0.40),
    "B": (0.20, 0.50),
    "C": (0.25, 0.55),
    "D": (0.30, 0.60),
}


# ============================================================
# Frequency masks
# ============================================================

def make_frequency_masks(h, w, lf_th, mf_th, device):

    fy = torch.arange(h, device=device).float()
    fx = torch.arange(w, device=device).float()

    yy, xx = torch.meshgrid(fy, fx, indexing="ij")

    r = torch.sqrt(
        (yy / (h - 1)) ** 2 +
        (xx / (w - 1)) ** 2
    ) / np.sqrt(2.0)

    lf = r <= lf_th
    mf = (r > lf_th) & (r <= mf_th)
    hf = r > mf_th

    return lf, mf, hf


# ============================================================
# LRS feature extraction
# ============================================================

def extract_features(model, x):

    x = model.Conv2d_1a_3x3(x)
    x = model.Conv2d_2a_3x3(x)
    x = model.Conv2d_2b_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)

    x = model.Conv2d_3b_1x1(x)
    x = model.Conv2d_4a_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)

    x = model.Mixed_5b(x)
    shallow = x

    x = model.Mixed_5c(x)
    x = model.Mixed_5d(x)
    balanced = x

    x = model.Mixed_6a(x)
    x = model.Mixed_6b(x)
    x = model.Mixed_6c(x)
    x = model.Mixed_6d(x)
    x = model.Mixed_6e(x)
    deep = x

    return shallow, balanced, deep


# ============================================================
# Frequency statistics
# ============================================================

def analyze_frequency(feature, lf_th, mf_th):

    B, C, H, W = feature.shape

    coeff = dct_2d(feature)

    lf_mask, mf_mask, hf_mask = make_frequency_masks(
        H,
        W,
        lf_th,
        mf_th,
        feature.device
    )

    energy = coeff.pow(2)

    total = energy.sum()

    lf_energy = energy[..., lf_mask].sum()
    mf_energy = energy[..., mf_mask].sum()
    hf_energy = energy[..., hf_mask].sum()

    lf = (lf_energy / total).item()
    mf = (mf_energy / total).item()
    hf = (hf_energy / total).item()

    return {
        "H": H,
        "W": W,

        "LF": lf,
        "MF": mf,
        "HF": hf,

        "sum": lf + mf + hf,

        "LF_coeff": int(lf_mask.sum().item()),
        "MF_coeff": int(mf_mask.sum().item()),
        "HF_coeff": int(hf_mask.sum().item()),
    }


# ============================================================
# Load images
# ============================================================

def load_images():

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    images = []

    with open(INPUT_CSV, "r") as f:

        reader = csv.reader(f)

        for row in reader:

            if len(row) == 0:
                continue

            image_id = row[0].strip()

            if image_id.lower() in [
                "imageid",
                "filename",
                "image",
                "path"
            ]:
                continue

            candidates = [
                image_id,
                image_id + ".png",
                image_id + ".jpg",
                image_id + ".jpeg",
                image_id + ".JPEG",
                image_id + ".PNG",
            ]

            path = None

            for candidate in candidates:

                candidate_path = os.path.join(
                    INPUT_DIR,
                    candidate
                )

                if os.path.exists(candidate_path):

                    path = candidate_path
                    break

            if path is None:

                print("WARNING: missing:", image_id)
                continue

            img = Image.open(path).convert("RGB")
            img = transform(img)

            images.append(img)

    if len(images) == 0:

        raise RuntimeError(
            "No images were loaded. "
            "Check INPUT_CSV and INPUT_DIR."
        )

    return torch.stack(images)


# ============================================================
# Main
# ============================================================

def main():

    print("Device:", DEVICE)

    print("Loading Inception-v3...")

    model = pretrainedmodels.inceptionv3(
        num_classes=1000,
        pretrained="imagenet"
    )

    model = model.to(DEVICE)
    model.eval()

    images = load_images()

    print("Images loaded:", len(images))

    # --------------------------------------------------------
    # Statistics accumulator
    # --------------------------------------------------------

    results = {}

    for config_name in CONFIGS:

        results[config_name] = {
            "shallow": [],
            "balanced": [],
            "deep": []
        }

    # --------------------------------------------------------
    # Process batch-by-batch
    # --------------------------------------------------------

    batch_size = 2

    with torch.no_grad():

        for start in range(0, len(images), batch_size):

            end = min(start + batch_size, len(images))

            print(
                f"\rProcessing images "
                f"{start + 1}-{end}/{len(images)}",
                end="",
                flush=True
            )

            x = images[start:end].to(DEVICE)

            shallow, balanced, deep = extract_features(
                model,
                x
            )

            feature_dict = {
                "shallow": shallow,
                "balanced": balanced,
                "deep": deep
            }

            # ------------------------------------------------
            # Run all frequency configurations on this batch
            # ------------------------------------------------

            for config_name, (lf_th, mf_th) in CONFIGS.items():

                for feature_name, feature in feature_dict.items():

                    stat = analyze_frequency(
                        feature,
                        lf_th,
                        mf_th
                    )

                    results[config_name][feature_name].append(
                        stat
                    )

            # ------------------------------------------------
            # Explicitly release GPU tensors
            # ------------------------------------------------

            del shallow
            del balanced
            del deep
            del feature_dict
            del x

            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    print("\n")

    # ========================================================
    # Print results
    # ========================================================

    print("=" * 80)
    print("STEP 4 - FREQUENCY MASK ABLATION")
    print("=" * 80)

    for config_name, (lf_th, mf_th) in CONFIGS.items():

        print("\n" + "-" * 80)

        print(
            f"CONFIG {config_name} "
            f"(LF <= {lf_th:.2f}, "
            f"MF = {lf_th:.2f}-{mf_th:.2f}, "
            f"HF > {mf_th:.2f})"
        )

        print("-" * 80)

        for feature_name in [
            "shallow",
            "balanced",
            "deep"
        ]:

            stats = results[
                config_name
            ][feature_name]

            lf = np.mean([
                s["LF"] for s in stats
            ])

            mf = np.mean([
                s["MF"] for s in stats
            ])

            hf = np.mean([
                s["HF"] for s in stats
            ])

            H = stats[0]["H"]
            W = stats[0]["W"]

            print(f"\n{feature_name.upper()}")

            print(
                f"Feature size : "
                f"{H} x {W}"
            )

            print(
                f"LF coeff     : "
                f"{stats[0]['LF_coeff']}"
            )

            print(
                f"MF coeff     : "
                f"{stats[0]['MF_coeff']}"
            )

            print(
                f"HF coeff     : "
                f"{stats[0]['HF_coeff']}"
            )

            print(
                f"LF energy    : "
                f"{lf:.6f} "
                f"({lf * 100:.2f}%)"
            )

            print(
                f"MF energy    : "
                f"{mf:.6f} "
                f"({mf * 100:.2f}%)"
            )

            print(
                f"HF energy    : "
                f"{hf:.6f} "
                f"({hf * 100:.2f}%)"
            )

            print(
                f"Energy sum   : "
                f"{lf + mf + hf:.6f}"
            )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()