import os
import sys
# Lấy đường dẫn của thư mục gốc project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from Base.corruptions import apply_corruption

def visualize_multiple_corruptions(csv_path, image_root, image_col, sample_idx, severity):
    # --- DANH SÁCH CÁC LOẠI NHIỄU CẦN HIỂN THỊ ---
    noises_to_show = [
        'elastic_transform',
        'brightness',
        'contrast',
        'gaussian_noise',
        'shot_noise',
        'impulse_noise'
    ]
    
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

    # --- 3. Áp dụng các loại nhiễu ---
    corrupted_tensors = []
    for noise_type in noises_to_show:
        print(f"Áp dụng nhiễu: '{noise_type}' với severity={severity}")
        # Dùng .clone() để đảm bảo mỗi lần áp dụng nhiễu đều từ ảnh gốc
        corrupted_img = apply_corruption(original_tensor.clone(), noise_type, severity)
        corrupted_tensors.append(corrupted_img)

    # --- 4. Hiển thị ảnh ---
    # Tạo một lưới 2x4 để chứa 1 ảnh gốc + 6 ảnh nhiễu
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten() # Chuyển mảng axes 2D thành 1D để dễ truy cập

    # Ảnh gốc (vị trí đầu tiên)
    axes[0].imshow(original_tensor.permute(1, 2, 0))
    axes[0].set_title("Ảnh gốc")
    axes[0].axis('off')

    # Hiển thị 6 ảnh bị nhiễu
    for i, (noise_type, corrupted_tensor) in enumerate(zip(noises_to_show, corrupted_tensors)):
        ax = axes[i + 1] # Bắt đầu từ vị trí thứ 2 (index 1)
        ax.imshow(corrupted_tensor.permute(1, 2, 0))
        ax.set_title(f"{noise_type}")
        ax.axis('off')
        
    # Ẩn các ô thừa trong lưới (lưới 2x4 có 8 ô, ta chỉ dùng 7)
    for i in range(len(noises_to_show) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout(pad=1.5) # Điều chỉnh khoảng cách giữa các ảnh
    plt.show()


if __name__ == "__main__":
    # ==========================================================================
    # --- CẤU HÌNH ---
    # Thay đổi đường dẫn này cho phù hợp với máy của bạn
    CSV_PATH = "/Users/admin/Working/Data/nih_14_structured/validate.csv"
    IMAGE_ROOT_DIR = "/Users/admin/Working/Data/nih_14_structured/images"
    IMAGE_COLUMN_NAME = 'image_id'

    SAMPLE_INDEX = 1 # Chọn một ảnh bất kỳ

    # Chọn mức độ nhiễu từ 1 đến 5 để áp dụng cho tất cả
    SEVERITY_LEVEL = 1
    # ==========================================================================

    visualize_multiple_corruptions(
        csv_path=CSV_PATH,
        image_root=IMAGE_ROOT_DIR,
        image_col=IMAGE_COLUMN_NAME,
        sample_idx=SAMPLE_INDEX,
        severity=SEVERITY_LEVEL
    )