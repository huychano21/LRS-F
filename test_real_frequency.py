import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
import pretrainedmodels
from PIL import Image
from torchvision import transforms
from function.dct import dct_2d


# =========================
# Config
# =========================
INPUT_CSV = "dataset/images.csv"
INPUT_DIR = "dataset/images"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Same masks as Step 2
LF_TH = 0.20
MF_TH = 0.50


# =========================
# DCT frequency masks
# =========================
def make_frequency_masks(h, w, device):
    fy = torch.arange(h, device=device).float()
    fx = torch.arange(w, device=device).float()

    yy, xx = torch.meshgrid(fy, fx, indexing="ij")

    # normalized radial frequency: [0, 1]
    r = torch.sqrt(
        (yy / (h - 1)) ** 2 +
        (xx / (w - 1)) ** 2
    ) / np.sqrt(2.0)

    lf = r <= LF_TH
    mf = (r > LF_TH) & (r <= MF_TH)
    hf = r > MF_TH

    return lf, mf, hf


# =========================
# Feature extraction
# =========================
def extract_features(model, x):
    """
    Inception-v3 feature locations corresponding to LRS:
      shallow  : after Mixed_5b
      balanced : after Mixed_5d
      deep     : after Mixed_6e
    """

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


# =========================
# Frequency energy
# =========================
def frequency_energy(feature):
    """
    feature: [B, C, H, W]

    DCT is applied independently to each channel.
    """

    B, C, H, W = feature.shape

    coeff = dct_2d(feature)

    lf_mask, mf_mask, hf_mask = make_frequency_masks(
        H, W, feature.device
    )

    # squared DCT coefficient magnitude
    energy = coeff.pow(2)

    total_energy = energy.sum()

    lf_energy = energy[..., lf_mask].sum()
    mf_energy = energy[..., mf_mask].sum()
    hf_energy = energy[..., hf_mask].sum()

    lf_ratio = (lf_energy / total_energy).item()
    mf_ratio = (mf_energy / total_energy).item()
    hf_ratio = (hf_energy / total_energy).item()

    return {
        "H": H,
        "W": W,
        "LF": lf_ratio,
        "MF": mf_ratio,
        "HF": hf_ratio,
        "sum": lf_ratio + mf_ratio + hf_ratio,
        "LF_coeff": int(lf_mask.sum().item()),
        "MF_coeff": int(mf_mask.sum().item()),
        "HF_coeff": int(hf_mask.sum().item()),
    }


# =========================
# Load images
# =========================
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

            # Skip header
            if image_id.lower() in ["imageid", "filename", "image", "path"]:
                continue

            # Try common extensions
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
                candidate_path = os.path.join(INPUT_DIR, candidate)

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
            "No images were loaded. Check INPUT_CSV and INPUT_DIR."
        )

    return torch.stack(images)


# =========================
# Main
# =========================
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

    results = {
        "shallow": [],
        "balanced": [],
        "deep": []
    }

    batch_size = 5

    with torch.no_grad():

        for start in range(0, len(images), batch_size):

            x = images[start:start + batch_size].to(DEVICE)

            shallow, balanced, deep = extract_features(model, x)

            for name, feat in [
                ("shallow", shallow),
                ("balanced", balanced),
                ("deep", deep)
            ]:

                stat = frequency_energy(feat)
                results[name].append(stat)

    print("\n" + "=" * 70)
    print("REAL INCEPTION-v3 FREQUENCY STATISTICS")
    print("=" * 70)

    for name in ["shallow", "balanced", "deep"]:

        stats = results[name]

        lf = np.mean([x["LF"] for x in stats])
        mf = np.mean([x["MF"] for x in stats])
        hf = np.mean([x["HF"] for x in stats])

        H = stats[0]["H"]
        W = stats[0]["W"]

        print(f"\n{name.upper()}")
        print(f"Feature size : {H} x {W}")
        print(f"LF coeff     : {stats[0]['LF_coeff']}")
        print(f"MF coeff     : {stats[0]['MF_coeff']}")
        print(f"HF coeff     : {stats[0]['HF_coeff']}")

        print(f"LF energy    : {lf:.6f} ({lf * 100:.2f}%)")
        print(f"MF energy    : {mf:.6f} ({mf * 100:.2f}%)")
        print(f"HF energy    : {hf:.6f} ({hf * 100:.2f}%)")
        print(f"Energy sum   : {lf + mf + hf:.6f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
