import os
import glob
import numpy as np
from PIL import Image
import argparse
import warnings

# Bỏ qua các cảnh báo chia cho 0 của numpy nếu có
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
except ImportError:
    print("Vui lòng cài đặt scikit-image để tính SSIM/PSNR:")
    print("pip install scikit-image")
    exit()

def load_image_as_numpy(path):
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Step 11 - Perceptual and Perturbation Statistics")
    parser.add_argument("--clean_dir", type=str, required=True, help="Thư mục chứa ảnh gốc (Clean)")
    parser.add_argument("--lrs_dir", type=str, required=True, help="Thư mục chứa ảnh LRS (Baseline)")
    parser.add_argument("--lrsf_dir", type=str, required=True, help="Thư mục chứa ảnh LRS-F")
    parser.add_argument("--num_images", type=int, default=1000, help="Số lượng ảnh cần đánh giá")
    args = parser.parse_args()

    # Quét toàn bộ file trong thư mục clean, loại bỏ file csv
    clean_images = glob.glob(os.path.join(args.clean_dir, "*.*"))
    clean_images = [f for f in clean_images if not f.endswith('.csv')]
    clean_images = sorted(clean_images)[:args.num_images]
    
    print("=" * 70)
    print("STEP 11 - PERCEPTUAL & PERTURBATION STATISTICS")
    print("=" * 70)
    print(f"Đang đánh giá trên {len(clean_images)} ảnh...")

    stats = {
        "lrs": {"l2": [], "linf": [], "psnr": [], "ssim": []},
        "lrsf": {"l2": [], "linf": [], "psnr": [], "ssim": []}
    }

    processed_count = 0

    for clean_path in clean_images:
        filename = os.path.basename(clean_path)
        base_name = os.path.splitext(filename)[0]

        # Tìm đúng file LRS (thử cả tên gốc và tên đuôi .png)
        lrs_exact = os.path.join(args.lrs_dir, filename)
        lrs_png = os.path.join(args.lrs_dir, base_name + ".png")
        lrs_path = lrs_exact if os.path.exists(lrs_exact) else lrs_png

        # Tìm đúng file LRS-F
        lrsf_exact = os.path.join(args.lrsf_dir, filename)
        lrsf_png = os.path.join(args.lrsf_dir, base_name + ".png")
        lrsf_path = lrsf_exact if os.path.exists(lrsf_exact) else lrsf_png

        # Nếu một trong 2 file không tồn tại, bỏ qua ảnh này
        if not os.path.exists(lrs_path) or not os.path.exists(lrsf_path):
            continue

        # Load ảnh
        img_clean = load_image_as_numpy(clean_path)
        img_lrs = load_image_as_numpy(lrs_path)
        img_lrsf = load_image_as_numpy(lrsf_path)

        # Cân bằng kích thước nếu cần
        if img_clean.shape != img_lrs.shape:
            img_clean_pil = Image.open(clean_path).convert("RGB").resize((img_lrs.shape[1], img_lrs.shape[0]))
            img_clean = np.array(img_clean_pil).astype(np.float32)

        # Tính L2 và Linf (thang 0-255)
        diff_lrs = img_lrs - img_clean
        diff_lrsf = img_lrsf - img_clean

        stats["lrs"]["linf"].append(np.max(np.abs(diff_lrs)))
        stats["lrsf"]["linf"].append(np.max(np.abs(diff_lrsf)))

        stats["lrs"]["l2"].append(np.linalg.norm(diff_lrs.flatten()) / np.sqrt(diff_lrs.size))
        stats["lrsf"]["l2"].append(np.linalg.norm(diff_lrsf.flatten()) / np.sqrt(diff_lrsf.size))

        # Tính PSNR và SSIM
        val_psnr_lrs = psnr(img_clean, img_lrs, data_range=255.0)
        val_ssim_lrs = ssim(img_clean, img_lrs, data_range=255.0, channel_axis=2)
        
        val_psnr_lrsf = psnr(img_clean, img_lrsf, data_range=255.0)
        val_ssim_lrsf = ssim(img_clean, img_lrsf, data_range=255.0, channel_axis=2)

        stats["lrs"]["psnr"].append(val_psnr_lrs)
        stats["lrs"]["ssim"].append(val_ssim_lrs)
        stats["lrsf"]["psnr"].append(val_psnr_lrsf)
        stats["lrsf"]["ssim"].append(val_ssim_lrsf)
        
        processed_count += 1
        if processed_count % 100 == 0:
            print(f" Đã xử lý {processed_count}/{len(clean_images)} ảnh...")

    if processed_count == 0:
        print("\n[LỖI] Không tìm thấy ảnh adversarial đối chiếu nào! Hãy kiểm tra lại thư mục lrs_dir và lrsf_dir.")
        return

    print("=" * 70)
    print("KẾT QUẢ ĐÁNH GIÁ TRUNG BÌNH")
    print("=" * 70)
    print(f"{'Metric':<15} | {'MI-LRS':<20} | {'MI-LRSF':<20}")
    print("-" * 60)
    
    metrics = ["linf", "l2", "psnr", "ssim"]
    labels = ["L_inf (0-255)", "L2 (RMSE)", "PSNR (dB)", "SSIM"]
    
    for metric, label in zip(metrics, labels):
        val_lrs = np.mean(stats["lrs"][metric])
        val_lrsf = np.mean(stats["lrsf"][metric])
        print(f"{label:<15} | {val_lrs:<20.4f} | {val_lrsf:<20.4f}")
        
    print("=" * 70)
    print("STEP 11 STATUS: HOÀN THÀNH")

if __name__ == "__main__":
    main()