import os
import sys
# Lấy đường dẫn của thư mục gốc project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os

# Import các lớp Dataset và config từ project của bạn
from core.configs import cfg
from Base.multilabel_loader import CleanSingleDomainDataset, collate_fn_skip_none
from Base.corruptions import apply_corruption

def calculate_distribution_stats(data_loader, apply_corruption_config=None):
    """
    Tính toán mean và std của một bộ dữ liệu được cung cấp bởi DataLoader.

    Args:
        data_loader (DataLoader): DataLoader cung cấp các batch ảnh.
        apply_corruption_config (dict, optional): Dict chứa thông tin để áp dụng nhiễu.
                                                 Ví dụ: {'name': 'gaussian_noise', 'severity': 1.0}
    
    Returns:
        tuple: (mean, std), mỗi cái là một tensor torch [R, G, B].
    """
    
    channels_sum = 0.
    channels_squared_sum = 0.
    num_pixels = 0

    pbar = tqdm(data_loader, desc=f"Calculating Stats")
    
    for data_package in pbar:
        if not data_package['image'].numel():
            continue
            
        # Lấy batch ảnh (dữ liệu này nằm trong khoảng [0, 1])
        images = data_package['image']

        # Nếu có yêu cầu, áp dụng nhiễu
        if apply_corruption_config:
            images = apply_corruption(images, 
                                      apply_corruption_config['name'],
                                      apply_corruption_config['severity'])
            pbar.set_description(f"Calculating Stats for {apply_corruption_config['name']}")

        # Lấy kích thước batch, channels, height, width
        b, c, h, w = images.shape
        
        # Tính tổng và tổng bình phương trên batch và cộng dồn
        # sum() trên các chiều không gian (H, W) và batch (B)
        channels_sum += torch.sum(images, dim=[0, 2, 3])
        channels_squared_sum += torch.sum(images**2, dim=[0, 2, 3])
        
        # Cộng dồn tổng số pixel
        num_pixels += b * h * w

    # Tính mean và std cuối cùng
    mean = channels_sum / num_pixels
    std = (channels_squared_sum / num_pixels - mean**2).sqrt()

    return mean, std


if __name__ == '__main__':
    # ==========================================================================
    # --- CẤU HÌNH ---
    # Load config file chính để lấy các đường dẫn
    CONFIG_FILE = 'configs/adapter/zero_shot.yaml' 
    cfg.merge_from_file(CONFIG_FILE)
    
    # Định nghĩa transform KHÔNG CÓ NORMALIZE
    transform_no_norm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), # Chuyển ảnh về tensor [0, 1]
    ])
    # ==========================================================================

    print("\n--- Analyzing CheXpert (Source Domain) ---")
    chexpert_cfg = cfg.clone(); chexpert_cfg.defrost()
    chexpert_cfg.DATASET.BASE_DOMAIN.PATH = "/home/ngoto/Working/Data/CheXpert-v1.0-small" 
    chexpert_cfg.DATASET.BASE_DOMAIN.CSV = "valid_final.csv"
    chexpert_cfg.DATASET.BASE_DOMAIN.IMAGE_DIR = ""
    chexpert_cfg.freeze()
    
    chexpert_dataset = CleanSingleDomainDataset(chexpert_cfg, transform=transform_no_norm)
    chexpert_loader = DataLoader(chexpert_dataset, batch_size=cfg.TEST.BATCH_SIZE, num_workers=cfg.LOADER.NUM_WORKS, collate_fn=collate_fn_skip_none)
    
    chexpert_mean, chexpert_std = calculate_distribution_stats(chexpert_loader)
    print(f"CheXpert Mean: {chexpert_mean.tolist()}")
    print(f"CheXpert Std:  {chexpert_std.tolist()}")

    
    # NIH Clearn
    print("\n--- Analyzing NIH-14 (Target Domain, Clean) ---")
    # NIH-14 đã được định nghĩa là BASE_DOMAIN trong config chính của bạn
    nih_dataset_clean = CleanSingleDomainDataset(cfg, transform=transform_no_norm)
    nih_loader_clean = DataLoader(nih_dataset_clean, batch_size=cfg.TEST.BATCH_SIZE, num_workers=cfg.LOADER.NUM_WORKS, collate_fn=collate_fn_skip_none)
    
    nih_clean_mean, nih_clean_std = calculate_distribution_stats(nih_loader_clean)
    print(f"NIH-14 Clean Mean: {nih_clean_mean.tolist()}")
    print(f"NIH-14 Clean Std:  {nih_clean_std.tolist()}")
    
    
    # Phân tích NIH-14 (Target Domain, Có nhiễu) ---
    print("\n--- Analyzing NIH-14 (Target Domain, Corrupted) ---")
    # Vẫn dùng dataloader sạch, nhưng truyền config nhiễu vào hàm tính toán
    corruption_config_to_test = {
        'name': 'gaussian_noise',
        'severity': 5.0
    }
    
    nih_corrupted_mean, nih_corrupted_std = calculate_distribution_stats(
        nih_loader_clean, 
        apply_corruption_config=corruption_config_to_test
    )
    print(f"NIH-14 Corrupted ({corruption_config_to_test['name']}, sev={corruption_config_to_test['severity']}):")
    print(f"  Mean: {nih_corrupted_mean.tolist()}")
    print(f"  Std:  {nih_corrupted_std.tolist()}")
    
    
    
    
    
    
    
    
    # # --- 2. Phân tích NIH-14 (Target Domain, Sạch) ---
    # print("\n--- Analyzing NIH-14 (Target Domain, Clean) ---")
    # # NIH-14 đã được định nghĩa là BASE_DOMAIN trong config chính của bạn
    # nih_dataset_clean = CleanSingleDomainDataset(cfg, transform=transform_no_norm)
    # nih_loader_clean = DataLoader(nih_dataset_clean, batch_size=cfg.TEST.BATCH_SIZE, num_workers=cfg.LOADER.NUM_WORKS, collate_fn=collate_fn_skip_none)
    
    # nih_clean_mean, nih_clean_std = calculate_distribution_stats(nih_loader_clean)
    # print(f"NIH-14 Clean Mean: {nih_clean_mean.tolist()}")
    # print(f"NIH-14 Clean Std:  {nih_clean_std.tolist()}")

    # # --- 3. Phân tích NIH-14 (Target Domain, Có nhiễu) ---
    # print("\n--- Analyzing NIH-14 (Target Domain, Corrupted) ---")
    # # Vẫn dùng dataloader sạch, nhưng truyền config nhiễu vào hàm tính toán
    # corruption_config_to_test = {
    #     'name': 'gaussian_noise',
    #     'severity': 1.0
    # }
    
    # nih_corrupted_mean, nih_corrupted_std = calculate_distribution_stats(
    #     nih_loader_clean, 
    #     apply_corruption_config=corruption_config_to_test
    # )
    # print(f"NIH-14 Corrupted ({corruption_config_to_test['name']}, sev={corruption_config_to_test['severity']}):")
    # print(f"  Mean: {nih_corrupted_mean.tolist()}")
    # print(f"  Std:  {nih_corrupted_std.tolist()}")