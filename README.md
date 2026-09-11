# LRS-F: Exploring Low-Rank, Sparsity, and Frequency in Adversarial Attacks

Nghiên cứu mở rộng phương pháp LRS-Attack bằng cách tích hợp biến đổi miền tần số (2D-DCT) phân cấp theo độ sâu mạng.

## Cấu trúc thư mục chính
- `function/LRS.py`: Cài đặt hàm `multi_lrsf_inv3` kết hợp Low-Rank, Sparse và Depth-aware Frequency.
- `function/dct.py`: Biến đổi 2D-DCT, IDCT và trích xuất mặt nạ tần số (LF, MF, HF).
- `Attack/CNN/Inc-v3/MI-LRSF.py`: Kịch bản tấn công chính MI + LRS-F (hỗ trợ tinh chỉnh `--frequency_weight`).
- `Attack/CNN/Inc-v3/Frequency-MI-FGSM.py`: Tấn công độc lập chỉ dùng miền tần số.
- `Attack/CNN/Inc-v3/step9A_mechanistic_validation.py`: Kiểm chứng cơ chế toán học và khôi phục DCT.
- `Attack/CNN/Inc-v3/step9B_expert_gradient_validation.py`: Kiểm tra tính phân biệt gradient giữa các expert.
- `Attack/CNN/Inc-v3/step9C_optimization_dynamics_validation.py`: Theo dõi động lực học tối ưu (Loss, Grad L2) qua 10 vòng lặp.
- `Attack/CNN/Inc-v3/step11_perceptual_statistics.py`: Đo đạc chất lượng ảnh adversarial (SSIM, PSNR, L2, Linf).
- `verify_cnns.py`: Đo lường ASR trên 12 mô hình Black-box.

## Hướng dẫn chạy thử nghiệm

1. Tấn công MI + LRS-F:
   python Attack/CNN/Inc-v3/MI-LRSF.py --input_csv dataset/images.csv --input_dir dataset/images --output_dir Attack/outputs/incv3-LRSF-weight-0.05 --frequency_weight 0.05

2. Đánh giá ASR:
   python verify_cnns.py --adv_dir Attack/outputs/incv3-LRSF-weight-0.05 --input_csv dataset/images.csv --input_dir dataset/images

3. Đánh giá chất lượng thị giác (SSIM/PSNR):
   python Attack/CNN/Inc-v3/step11_perceptual_statistics.py --clean_dir dataset/images --lrs_dir Attack/outputs/incv3-MI-FGSM-lrs --lrsf_dir Attack/outputs/incv3-LRSF-weight-0.05
