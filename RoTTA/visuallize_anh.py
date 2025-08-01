import torch
import pandas as pd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Import các hàm từ project của bạn
# Giả sử script này nằm ở thư mục gốc của project
from core.data.corruptions import apply_corruption

def visualize_single_corruption(csv_path, image_root, image_col, labels_list, sample_idx, corruption_name, severity):
    """
    Tải một ảnh, áp dụng nhiễu, và hiển thị kết quả.
    """
    # --- 1. Tải ảnh và nhãn gốc ---
    try:
        df = pd.read_csv(csv_path)
        row = df.iloc[sample_idx]
        img_name = str(row[image_col])
        if not os.path.isabs(img_name):
            img_path = os.path.join(image_root, img_name)
        else:
            img_path = img_name
        
        original_pil_image = Image.open(img_path).convert('RGB')
        labels = row[labels_list].to_dict()
    except Exception as e:
        print(f"Lỗi khi tải ảnh mẫu: {e}")
        return

    print(f"--- Thông tin ảnh mẫu ---")
    print(f"Đường dẫn: {img_path}")
    print(f"Các nhãn: {labels}")

    # --- 2. Định nghĩa các bước tiền xử lý ---
    # Transform này giống hệt trong Dataloader của bạn
    base_transform_to_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), # Chuyển ảnh về tensor [0, 1]
    ])
    
    # Hàm để hiển thị (chuyển tensor đã normalize về lại dạng có thể xem)
    def denormalize(tensor):
        # Giả sử mean và std chuẩn
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean

    # --- 3. Xử lý và áp dụng nhiễu ---
    # Ảnh gốc dưới dạng tensor, chưa normalize
    original_tensor_unnormalized = base_transform_to_tensor(original_pil_image)

    # Áp dụng nhiễu. Hàm apply_corruption nhận tensor [0, 1]
    corrupted_tensor_unnormalized = apply_corruption(
        original_tensor_unnormalized.clone(), # Dùng clone để không thay đổi tensor gốc
        corruption_name, 
        severity
    )
    
    # --- 4. In thông số của các tensor ---
    print("\n--- Phân tích Tensor (phạm vi [0, 1]) ---")
    print(f"Ảnh gốc:      Size={original_tensor_unnormalized.shape}, Min={original_tensor_unnormalized.min():.4f}, Max={original_tensor_unnormalized.max():.4f}, Mean={original_tensor_unnormalized.mean():.4f}")
    print(f"Ảnh bị nhiễu:   Size={corrupted_tensor_unnormalized.shape}, Min={corrupted_tensor_unnormalized.min():.4f}, Max={corrupted_tensor_unnormalized.max():.4f}, Mean={corrupted_tensor_unnormalized.mean():.4f}")
    
    # --- 5. Hiển thị ảnh ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Ảnh gốc
    axes[0].imshow(original_tensor_unnormalized.permute(1, 2, 0))
    axes[0].set_title("Ảnh gốc (chưa Normalize)")
    axes[0].axis('off')

    # Ảnh bị nhiễu
    axes[1].imshow(corrupted_tensor_unnormalized.permute(1, 2, 0))
    axes[1].set_title(f"Ảnh sau khi áp dụng '{corruption_name}' (severity={severity})")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ==========================================================================
    # --- CẤU HÌNH ---
    # Hãy thay đổi các giá trị này cho phù hợp
    
    # 1. Đường dẫn đến file CSV và thư mục ảnh
    CSV_PATH = "/home/ngoto/Working/Data/MixData/nih_14_structured/validate_reordered.csv"
    IMAGE_ROOT_DIR = "/home/ngoto/Working/Data/MixData/nih_14_structured/images"
    IMAGE_COLUMN_NAME = 'image_id' # Hoặc 'Path' tùy file CSV

    # 2. Danh sách các nhãn (phải khớp với cột trong CSV)
    LABELS = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 
        'Pleural Effusion', 'Pneumothorax' # Ví dụ với 5 nhãn
    ]

    # 3. Chọn một ảnh mẫu để xem (thay đổi chỉ số này)
    SAMPLE_INDEX = 10 

    # 4. Chọn loại nhiễu và mức độ để kiểm tra
    CORRUPTION_TO_TEST = 'gaussian_noise'
    SEVERITY_LEVEL = 1
    # ==========================================================================

    visualize_single_corruption(
        csv_path=CSV_PATH,
        image_root=IMAGE_ROOT_DIR,
        image_col=IMAGE_COLUMN_NAME,
        labels_list=LABELS,
        sample_idx=SAMPLE_INDEX,
        corruption_name=CORRUPTION_TO_TEST,
        severity=SEVERITY_LEVEL
    )