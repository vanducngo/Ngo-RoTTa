import os
import sys
# Lấy đường dẫn của thư mục gốc project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

# Import các thành phần từ project của bạn
from core.configs import cfg
from Base.corruptions import apply_corruption
from Base.multilabel_loader import CleanSingleDomainDataset, collate_fn_skip_none

def calculate_distribution_stats_for_corruption(base_data_loader, corruption_name, severity):
    """
    Tính toán mean và std cho MỘT loại nhiễu cụ thể trên toàn bộ dataset.
    """
    channels_sum = 0.
    channels_squared_sum = 0.
    num_pixels = 0

    pbar = tqdm(base_data_loader, desc=f"Analyzing '{corruption_name}'", leave=False)
    
    for data_package in pbar:
        if not data_package['image'].numel():
            continue
            
        clean_images = data_package['image']
        
        # Áp dụng một loại nhiễu duy nhất
        corrupted_images = apply_corruption(clean_images, corruption_name, severity)

        b, c, h, w = corrupted_images.shape
        
        channels_sum += torch.sum(corrupted_images, dim=[0, 2, 3])
        channels_squared_sum += torch.sum(corrupted_images**2, dim=[0, 2, 3])
        num_pixels += b * h * w

    mean = channels_sum / num_pixels
    std = (channels_squared_sum / num_pixels - mean**2).sqrt()

    return mean.tolist(), std.tolist()


if __name__ == '__main__':
    # ==========================================================================
    # --- CẤU HÌNH ---
    CONFIG_FILE = 'configs/adapter/zero_shot.yaml' 
    cfg.merge_from_file(CONFIG_FILE)
    cfg.freeze()

    SEVERITY_TO_ANALYZE = 5.0 # Mức độ nhiễu bạn muốn phân tích
    
    # Transform KHÔNG CÓ NORMALIZE
    transform_no_norm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), # Chuyển ảnh về tensor [0, 1]
    ])
    # ==========================================================================

    print("\n--- Analyzing Distribution for Each Corruption Type ---")
    print(f"Base Dataset: NIH-14")
    print(f"Severity Level: {SEVERITY_TO_ANALYZE}")

    # --- 1. Tạo DataLoader cho dữ liệu sạch (chỉ cần tạo một lần) ---
    clean_dataset = CleanSingleDomainDataset(cfg, transform=transform_no_norm)
    clean_loader = DataLoader(
        clean_dataset, 
        batch_size=cfg.TEST.BATCH_SIZE, 
        num_workers=cfg.LOADER.NUM_WORKS, 
        collate_fn=collate_fn_skip_none
    )

    # --- 2. Lấy danh sách các loại nhiễu và dữ liệu sạch gốc ---
    corruptions_list = cfg.DATASET.TEST_CORRUPTIONS
    # Tính toán cho dữ liệu sạch trước
    print("\nCalculating stats for CLEAN data...")
    clean_mean, clean_std = calculate_distribution_stats_for_corruption(
        clean_loader, 'none', SEVERITY_TO_ANALYZE
    )
    
    # --- 3. Lặp qua từng loại nhiễu và tính toán ---
    results = [{'Corruption': 'Clean (Base)', 'Mean': f"{clean_mean[0]:.4f}", 'Std': f"{clean_std[0]:.4f}"}]

    for corruption in corruptions_list:
        mean, std = calculate_distribution_stats_for_corruption(
            clean_loader, corruption, SEVERITY_TO_ANALYZE
        )
        results.append({
            'Corruption': corruption,
            'Mean': f"{mean[0]:.4f}",
            'Std': f"{std[0]:.4f}"
        })
    
    # --- 4. In kết quả ra bảng ---
    results_df = pd.DataFrame(results)
    
    print("\n--- Distribution Stats per Corruption Type ---")
    print(results_df.to_string(index=False))