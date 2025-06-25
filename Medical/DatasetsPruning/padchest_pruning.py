import pandas as pd
import os
import shutil
from tqdm import tqdm
import ast  # Dùng để chuyển chuỗi thành list một cách an toàn

# ==============================================================================
# ĐỊNH NGHĨA CÁC THAM SỐ VÀ HẰNG SỐ
# ==============================================================================

# Cấu hình đường dẫn (BẠN CẦN THAY ĐỔI CÁC ĐƯỜNG DẪN NÀY)
PADCHEST_ROOT_PATH = "/Users/admin/Working/Data/PadChest-Origin"  # Thư mục gốc chứa ảnh và file .csv
OUTPUT_PATH = "/Users/admin/Working/Data/PadChestPruning"

# Định nghĩa bộ nhãn chúng ta muốn giữ lại (dạng chữ thường cho PadChest)
DISEASES_TO_KEEP_LOWER = [
    'atelectasis',
    'cardiomegaly',
    'consolidation',
    'pleural effusion',
    'pneumothorax'
]
# Nhãn "bình thường" trong PadChest
NORMAL_LABEL = 'normal'

# Tên cột nhãn cuối cùng (chuẩn hóa)
FINAL_LABEL_SET_MAPPED = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Pleural Effusion', 'Pneumothorax']

# ==============================================================================
# HÀM XỬ LÝ CHÍNH
# ==============================================================================

def preprocess_and_filter_padchest():
    """
    Lọc và sao chép dữ liệu cho bộ dữ liệu PadChest.
    """
    print("--- Starting preprocessing for PadChest set ---")
    
    # 1. Tạo đường dẫn
    source_csv_path = os.path.join(PADCHEST_ROOT_PATH, 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv')
    source_image_dir = os.path.join(PADCHEST_ROOT_PATH, 'images-224', 'images-224')
    
    if not os.path.exists(source_csv_path) or not os.path.exists(source_image_dir):
        print(f"Error: Source CSV or image directory not found.")
        return

    target_dir = os.path.join(OUTPUT_PATH)
    target_image_dir = os.path.join(target_dir, 'images')
    target_csv_path = os.path.join(target_dir, 'refined_padchest_labels.csv')

    os.makedirs(target_image_dir, exist_ok=True)

    # 2. Đọc và lọc DataFrame gốc
    print("Reading and filtering original padchest_labels.csv...")
    df_raw = pd.read_csv(source_csv_path)
    
    # Xử lý cột 'Labels' để thay thế nan bằng danh sách rỗng
    print("Handling 'Labels' column and converting to list...")
    df_raw['Labels'] = df_raw['Labels'].fillna('[]')  # Thay nan bằng chuỗi rỗng []
    df_raw['Labels_List'] = df_raw['Labels'].apply(lambda x: [label.strip() for label in ast.literal_eval(x) if pd.notna(x)] if pd.notna(x) else [])

    # Điều kiện lọc: giữ lại một hàng nếu list nhãn của nó chứa 'normal'
    # hoặc bất kỳ bệnh nào trong DISEASES_TO_KEEP_LOWER
    def filter_condition(label_list):
        return NORMAL_LABEL in [label.strip() for label in label_list] or any(disease in [label.strip() for label in label_list] for disease in DISEASES_TO_KEEP_LOWER)
        
    df_filtered = df_raw[df_raw['Labels_List'].apply(filter_condition)].copy()
    
    print(f"Original number of records: {len(df_raw)}")
    print(f"Number of records after filtering: {len(df_filtered)}")

    # 3. Tạo các cột nhãn nhị phân
    print("Creating one-hot encoded labels...")
    
    # Khởi tạo DataFrame mới chỉ với cột image_id và nhãn
    df_final = pd.DataFrame()
    df_final['image_id'] = df_filtered['ImageID']
    
    # Khởi tạo các cột one-hot với giá trị mặc định 0
    for label in FINAL_LABEL_SET_MAPPED:
        df_final[label] = 0
    
    # Lớp 'No Finding'
    df_final['No Finding'] = df_filtered['Labels_List'].apply(lambda x: 1 if NORMAL_LABEL in [label.strip() for label in x] and all(d not in [label.strip() for label in x] for d in DISEASES_TO_KEEP_LOWER) else 0)
    
    # Các lớp bệnh lý
    for disease_lower, disease_upper in zip(DISEASES_TO_KEEP_LOWER, FINAL_LABEL_SET_MAPPED[1:]):
        df_final[disease_upper] = df_filtered['Labels_List'].apply(lambda x: 1 if disease_lower in [label.strip() for label in x] else 0)

    # 4. Sao chép các file ảnh đã lọc và loại bỏ image_id không có ảnh
    print("Copying filtered image files and verifying existence...")
    num_copied = 0
    num_skipped = 0
    valid_image_ids = []
    
    for image_id in tqdm(df_final['image_id'], desc="Copying images"):
        source_image_path = os.path.join(source_image_dir, image_id)
        target_image_path = os.path.join(target_image_dir, image_id)
        
        try:
            if os.path.exists(source_image_path):
                if not os.path.exists(target_image_path):
                    shutil.copy2(source_image_path, target_image_path)
                num_copied += 1
                valid_image_ids.append(image_id)
            else:
                print(f"Warning: Source image not found, skipping: {source_image_path}")
                num_skipped += 1
        except Exception as e:
            print(f"Error copying {image_id}: {str(e)}")
            num_skipped += 1

    print(f"Finished copying. Copied: {num_copied}, Skipped: {num_skipped}")

    # Loại bỏ các image_id không có ảnh tương ứng
    df_final = df_final[df_final['image_id'].isin(valid_image_ids)].reset_index(drop=True)

    # 5. Lưu DataFrame đã lọc
    print(f"Saving new CSV to {target_csv_path}")
    df_final.to_csv(target_csv_path, index=False)
    
    print(f"--- Preprocessing for PadChest set completed! ---")
    print(f"Final number of records in CSV: {len(df_final)}")

if __name__ == "__main__":
    preprocess_and_filter_padchest()