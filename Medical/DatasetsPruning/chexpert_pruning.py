import pandas as pd
import os
import shutil
from tqdm import tqdm

# ==============================================================================
# ĐỊNH NGHĨA CÁC THAM SỐ VÀ HẰNG SỐ
# ==============================================================================

# Cấu hình đường dẫn (BẠN CẦN THAY ĐỔI CÁC ĐƯỜNG DẪN NÀY)
CHEXPERT_ROOT_PATH = "/path/to/your/CheXpert-v1.0-small"
OUTPUT_PATH = "/path/to/your/output/RefinedCheXpert"

# Định nghĩa bộ nhãn chúng ta muốn giữ lại
# Đây là các cột bệnh lý cần kiểm tra
DISEASES_TO_KEEP = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Pleural Effusion',
    'Pneumothorax'
]
# Toàn bộ các cột cuối cùng, bao gồm cả 'No Finding'
FINAL_LABEL_SET = ['No Finding'] + DISEASES_TO_KEEP


# ==============================================================================
# HÀM XỬ LÝ CHÍNH
# ==============================================================================

def preprocess_and_filter_chexpert(mode='train'):
    """
    Lọc và sao chép dữ liệu cho tập train hoặc valid của CheXpert.
    
    Args:
        mode (str): 'train' hoặc 'valid'.
    """
    print(f"--- Starting preprocessing for '{mode}' set ---")
    
    # 1. Tạo đường dẫn
    source_csv_path = os.path.join(CHEXPERT_ROOT_PATH, f'{mode}.csv')
    if not os.path.exists(source_csv_path):
        print(f"Error: Source CSV not found at {source_csv_path}")
        return

    target_dir = os.path.join(OUTPUT_PATH)
    target_csv_path = os.path.join(target_dir, f'refined_{mode}.csv')

    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(target_dir, exist_ok=True)

    # 2. Đọc DataFrame gốc
    print(f"Reading original {mode}.csv...")
    df_raw = pd.read_csv(source_csv_path)
    print(f"Original number of records: {len(df_raw)}")

    # 3. Lọc DataFrame
    # Điều kiện để giữ lại một hàng:
    # - Hoặc cột 'No Finding' có giá trị dương (1.0).
    # - Hoặc ít nhất một trong các cột bệnh lý trong DISEASES_TO_KEEP có giá trị dương (1.0).
    # Chúng ta cũng có thể xem xét giá trị không chắc chắn (-1.0) là một dạng "có thể có bệnh"
    
    print("Filtering records...")
    # Tạo một điều kiện lọc (mask)
    # df_raw[DISEASES_TO_KEEP] == 1.0 trả về một DataFrame boolean
    # .any(axis=1) kiểm tra xem có bất kỳ giá trị True nào trên mỗi hàng không
    condition = ((df_raw['No Finding'] == 1.0) | 
                 (df_raw[DISEASES_TO_KEEP] == 1.0).any(axis=1) |
                 (df_raw[DISEASES_TO_KEEP] == -1.0).any(axis=1))
                 
    df_filtered = df_raw[condition].copy()
    
    # Chỉ giữ lại các cột cần thiết cho file CSV mới
    columns_to_save = ['Path'] + FINAL_LABEL_SET
    df_final = df_filtered[columns_to_save]
    
    print(f"Number of records after filtering: {len(df_final)}")

    # 4. Sao chép các file ảnh đã lọc
    print("Copying filtered image files...")
    num_copied = 0
    num_skipped = 0
    for index, row in tqdm(df_final.iterrows(), total=len(df_final), desc=f"Copying {mode} images"):
        source_image_path = os.path.join(CHEXPERT_ROOT_PATH, row['Path'])
        
        # Tạo đường dẫn đích, giữ nguyên cấu trúc thư mục con
        relative_path = os.path.relpath(source_image_path, CHEXPERT_ROOT_PATH)
        target_image_path = os.path.join(target_dir, relative_path)
        
        # Tạo thư mục cha nếu chưa có
        os.makedirs(os.path.dirname(target_image_path), exist_ok=True)
        
        try:
            if not os.path.exists(target_image_path):
                shutil.copy2(source_image_path, target_image_path)
            num_copied += 1
        except FileNotFoundError:
            print(f"Warning: Source image not found, skipping: {source_image_path}")
            num_skipped += 1
            # Có thể xóa hàng này khỏi df_final nếu file ảnh không tồn tại
            # df_final.drop(index, inplace=True)
            
    print(f"Finished copying. Copied: {num_copied}, Skipped: {num_skipped}")

    # 5. Lưu DataFrame đã lọc
    print(f"Saving new CSV to {target_csv_path}")
    # Cập nhật lại cột 'Path' để nó không còn chứa đường dẫn gốc
    df_final['Path'] = df_final['Path'].apply(lambda p: os.path.relpath(os.path.join(CHEXPERT_ROOT_PATH, p), target_dir))
    df_final.to_csv(target_csv_path, index=False)
    
    print(f"--- Preprocessing for '{mode}' set completed! ---\n")


# ==============================================================================
# HÀM MAIN ĐỂ CHẠY
# ==============================================================================

if __name__ == "__main__":
    print("===== Starting CheXpert Dataset Refinement Process =====")
    
    # Xử lý tập train
    preprocess_and_filter_chexpert(mode='train')
    
    # Xử lý tập valid (dùng làm tập test)
    preprocess_and_filter_chexpert(mode='valid')
    
    print("===== All tasks completed! =====")
    print(f"Refined dataset is now available at: {OUTPUT_PATH}")