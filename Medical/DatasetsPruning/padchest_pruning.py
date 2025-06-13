import pandas as pd
import os
import shutil
from tqdm import tqdm
import ast # Dùng để chuyển chuỗi thành list một cách an toàn

# ==============================================================================
# ĐỊNH NGHĨA CÁC THAM SỐ VÀ HẰNG SỐ
# ==============================================================================

# Cấu hình đường dẫn (BẠN CẦN THAY ĐỔI CÁC ĐƯỜNG DẪN NÀY)
PADCHEST_ROOT_PATH = "/path/to/your/padchest-small" # Thư mục gốc chứa ảnh và file .csv
OUTPUT_PATH = "/path/to/your/output/RefinedPadChest"

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
    source_csv_path = os.path.join(PADCHEST_ROOT_PATH, 'padchest_labels.csv')
    source_image_dir = os.path.join(PADCHEST_ROOT_PATH, 'images')
    
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
    
    # Chuyển cột 'Labels' từ string sang list
    # ast.literal_eval an toàn hơn eval()
    print("Converting 'Labels' column from string to list...")
    df_raw['Labels_List'] = df_raw['Labels'].apply(ast.literal_eval)

    # Điều kiện lọc: giữ lại một hàng nếu list nhãn của nó chứa 'normal'
    # hoặc bất kỳ bệnh nào trong DISEASES_TO_KEEP_LOWER
    def filter_condition(label_list):
        return NORMAL_LABEL in label_list or any(disease in label_list for disease in DISEASES_TO_KEEP_LOWER)
        
    df_filtered = df_raw[df_raw['Labels_List'].apply(filter_condition)].copy()
    
    print(f"Original number of records: {len(df_raw)}")
    print(f"Number of records after filtering: {len(df_filtered)}")

    # 3. Tạo các cột nhãn nhị phân
    print("Creating one-hot encoded labels...")
    
    # Lớp 'No Finding'
    df_filtered['No Finding'] = df_filtered['Labels_List'].apply(lambda x: 1 if NORMAL_LABEL in x else 0)
    
    # Các lớp bệnh lý
    for disease_lower, disease_upper in zip(DISEASES_TO_KEEP_LOWER, FINAL_LABEL_SET_MAPPED[1:]):
        df_filtered[disease_upper] = df_filtered['Labels_List'].apply(lambda x: 1 if disease_lower in x else 0)

    # Chỉ giữ lại các cột cần thiết
    columns_to_save = ['ImageID'] + FINAL_LABEL_SET_MAPPED
    df_final = df_filtered[columns_to_save].rename(columns={'ImageID': 'image_id'})

    # 4. Sao chép các file ảnh đã lọc
    print("Copying filtered image files...")
    num_copied = 0
    num_skipped = 0
    for image_id in tqdm(df_final['image_id'], desc="Copying images"):
        source_image_path = os.path.join(source_image_dir, image_id)
        target_image_path = os.path.join(target_image_dir, image_id)
        
        try:
            if not os.path.exists(target_image_path):
                shutil.copy2(source_image_path, target_image_path)
            num_copied += 1
        except FileNotFoundError:
            print(f"Warning: Source image not found, skipping: {source_image_path}")
            num_skipped += 1
            df_final = df_final[df_final['image_id'] != image_id]
            
    print(f"Finished copying. Copied: {num_copied}, Skipped: {num_skipped}")

    # 5. Lưu DataFrame đã lọc
    print(f"Saving new CSV to {target_csv_path}")
    df_final.to_csv(target_csv_path, index=False)
    
    print(f"--- Preprocessing for PadChest set completed! ---\n")

if __name__ == "__main__":
    preprocess_and_filter_padchest()