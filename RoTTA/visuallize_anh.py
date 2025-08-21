import os
import sys
# Lấy đường dẫn của thư mục gốc project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


import torch
import pandas as pd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from Base.corruptions import apply_corruption

# Thêm thư mục gốc vào path để có thể import các module
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import các hàm từ project của bạn
from Base.corruptions import snow, motion_blur, _linear_interpolate

def visualize_snow_corruption(csv_path, image_root, image_col, sample_idx, severity):
    noise = 'defocus_blur'
    # --- 1. Tải ảnh gốc ---
    try:
        df = pd.read_csv(csv_path)
        row = df.iloc[sample_idx]
        img_name = str(row[image_col])
        if not os.path.isabs(img_name):
            img_path = os.path.join(image_root, img_name)
        else:
            img_path = img_name
        
        original_pil_image = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Lỗi khi tải ảnh mẫu: {e}")
        return

    print(f"--- Đang visualize ảnh: {img_name} ---")

    # --- 2. Tiền xử lý ---
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), # Chuyển ảnh về tensor [0, 1]
    ])
    original_tensor = base_transform(original_pil_image)
    corrupted_tensor = apply_corruption(original_tensor, noise, 5)

    # # --- 3. Áp dụng nhiễu và lưu lại các bước trung gian ---
    # # Tái tạo lại logic bên trong hàm snow để có thể visualize mask
    # alpha_levels = [0, 0.1, 0.15, 0.2, 0.3, 0.4] 
    # blur_sev_levels = [0, 0.4, 0.6, 0.7, 0.8, 1.0]
    # brightness_levels = [1.0, 1.8, 2.7, 3.6, 4.5, 5.5]
    
    # alpha = _linear_interpolate(severity, alpha_levels)
    # blur_sev = _linear_interpolate(severity, blur_sev_levels)
    # brightness_factor = _linear_interpolate(severity, brightness_levels)

    # c, h, w = original_tensor.shape
    # snow_pattern_1ch = torch.rand(1, h, w, device=original_tensor.device)
    # snow_pattern_3ch = snow_pattern_1ch.repeat(c, 1, 1)
    
    # motion_blur_severity = blur_sev * 4 + 1
    # snow_pattern_blurred = motion_blur(snow_pattern_3ch, severity=motion_blur_severity)
    
    # snow_mask = (snow_pattern_blurred - snow_pattern_blurred.min()) / (snow_pattern_blurred.max() - snow_pattern_blurred.min() + 1e-6)
    
    # snow_layer = torch.ones_like(original_tensor) * brightness_factor
    
    # corrupted_tensor = original_tensor * (1 - snow_mask * alpha) + snow_layer * snow_mask * alpha
    # corrupted_tensor = torch.clamp(corrupted_tensor, 0, 1)
    
    # --- 4. In thông số ---
    print("\n--- Phân tích Tensor (phạm vi [0, 1]) ---")
    print(f"Ảnh gốc:      Min={original_tensor.min():.4f}, Max={original_tensor.max():.4f}, Mean={original_tensor.mean():.4f}")
    print(f"Ảnh bị nhiễu:   Min={corrupted_tensor.min():.4f}, Max={corrupted_tensor.max():.4f}, Mean={corrupted_tensor.mean():.4f}")
    
    # --- 5. Hiển thị ảnh ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Ảnh gốc
    axes[0].imshow(original_tensor.permute(1, 2, 0))
    axes[0].set_title("Ảnh gốc")
    axes[0].axis('off')

    # Ảnh bị nhiễu
    axes[1].imshow(corrupted_tensor.permute(1, 2, 0))
    axes[1].set_title(f"Ảnh sau khi áp dụng '{noise}' (severity={severity})")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ==========================================================================
    # --- CẤU HÌNH ---
    
    CSV_PATH = "/home/ngoto/Working/Data/MixData/nih_14_structured/validate_reordered.csv"
    IMAGE_ROOT_DIR = "/home/ngoto/Working/Data/MixData/nih_14_structured/images"
    IMAGE_COLUMN_NAME = 'image_id'

    SAMPLE_INDEX = 1 # Chọn một ảnh bất kỳ

    # Chạy visualize cho nhiễu SNOW ở mức độ nặng nhất
    SEVERITY_LEVEL = 5.0
    # ==========================================================================

    visualize_snow_corruption(
        csv_path=CSV_PATH,
        image_root=IMAGE_ROOT_DIR,
        image_col=IMAGE_COLUMN_NAME,
        sample_idx=SAMPLE_INDEX,
        severity=SEVERITY_LEVEL
    )