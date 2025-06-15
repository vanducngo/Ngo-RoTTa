import pandas as pd
import os
import shutil
from tqdm import tqdm

# ==============================================================================
# ĐỊNH NGHĨA CÁC THAM SỐ VÀ HẰNG SỐ
# ==============================================================================

# Cấu hình đường dẫn (BẠN CẦN THAY ĐỔI CÁC ĐƯỜNG DẪN NÀY)
# This path should point to the directory containing 'train.csv', 'valid.csv', etc.
CHEXPERT_ROOT_PATH = "../datasets/CheXpert-v1.0-small"
# This is where the new, filtered dataset will be created.
OUTPUT_PATH = "../datasets/RefinedCheXpert"

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

    # The output directory will be the root for the new dataset
    target_dir = os.path.join(OUTPUT_PATH)
    target_csv_path = os.path.join(target_dir, f'refined_{mode}.csv')

    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(target_dir, exist_ok=True)

    # 2. Đọc DataFrame gốc
    print(f"Reading original {mode}.csv...")
    df_raw = pd.read_csv(source_csv_path)
    print(f"Original number of records: {len(df_raw)}")

    # 3. Lọc DataFrame
    # Giữ lại các ảnh 'No Finding' hoặc có một trong các bệnh lý quan tâm (kể cả không chắc chắn)
    print("Filtering records...")
    condition = ((df_raw['No Finding'] == 1.0) | 
                 (df_raw[DISEASES_TO_KEEP] == 1.0).any(axis=1) |
                 (df_raw[DISEASES_TO_KEEP] == -1.0).any(axis=1))
                 
    df_filtered = df_raw[condition].copy()
    
    # Chỉ giữ lại các cột cần thiết cho file CSV mới
    columns_to_save = ['Path'] + FINAL_LABEL_SET
    df_final = df_filtered[columns_to_save].reset_index(drop=True)
    
    print(f"Number of records after filtering: {len(df_final)}")

    # 4. Sao chép các file ảnh đã lọc
    print("Copying filtered image files...")
    
    # ==========================================================================
    # START OF FIX (SOLUTION 1)
    # The paths in the CSV (e.g., "CheXpert-v1.0-small/train/...") are relative
    # to the PARENT of CHEXPERT_ROOT_PATH. We get that parent directory here.
    dataset_base_dir = os.path.dirname(CHEXPERT_ROOT_PATH) # This will be "../datasets"
    # ==========================================================================
    
    num_copied = 0
    num_skipped = 0
    
    # We will collect valid paths to update the dataframe later
    new_paths = []
    indices_to_drop = []

    for index, row in tqdm(df_final.iterrows(), total=len(df_final), desc=f"Copying {mode} images"):
        path_from_csv = row['Path']
        
        # ==========================================================================
        # FIX: Construct the full source path correctly to avoid duplication.
        # os.path.join("../datasets", "CheXpert-v1.0-small/train/...")
        source_image_path = os.path.join(dataset_base_dir, path_from_csv)
        # ==========================================================================

        # Create a cleaner relative path for the new dataset structure.
        # Instead of 'RefinedCheXpert/CheXpert-v1.0-small/train/...', we'll have
        # 'RefinedCheXpert/train/...'
        relative_path = os.path.relpath(source_image_path, CHEXPERT_ROOT_PATH)
        target_image_path = os.path.join(target_dir, relative_path)
        
        # Create the parent directory (e.g., .../RefinedCheXpert/train/patientXXXXX/studyY)
        os.makedirs(os.path.dirname(target_image_path), exist_ok=True)
        
        if os.path.exists(source_image_path):
            try:
                if not os.path.exists(target_image_path):
                    shutil.copy2(source_image_path, target_image_path)
                num_copied += 1
                new_paths.append(relative_path) # Store the new, clean path
            except Exception as e:
                print(f"Warning: Could not copy file {source_image_path}. Error: {e}")
                num_skipped += 1
                indices_to_drop.append(index) # Mark this row for removal
        else:
            # This handles the case where the image listed in the CSV doesn't exist at all
            print(f"Warning: Source image not found, skipping: {source_image_path}")
            num_skipped += 1
            indices_to_drop.append(index) # Mark this row for removal
            
    print(f"Finished copying. Copied: {num_copied}, Skipped: {num_skipped}")

    # Remove rows from the DataFrame where the image file was missing or couldn't be copied
    if indices_to_drop:
        print(f"Removing {len(indices_to_drop)} records from CSV due to missing images.")
        df_final.drop(indices_to_drop, inplace=True)

    # 5. Lưu DataFrame đã lọc
    print(f"Saving new CSV to {target_csv_path}")

    # ==========================================================================
    # FIX: Update the 'Path' column to the new relative paths
    # This makes the new CSV self-contained within the OUTPUT_PATH directory.
    df_final['Path'] = new_paths
    # ==========================================================================
    
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
    print("The new CSV files inside that directory now use relative paths.")