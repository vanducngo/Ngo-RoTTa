import pandas as pd
import os
import shutil
from tqdm import tqdm

# ==============================================================================
# ĐỊNH NGHĨA CÁC THAM SỐ VÀ HẰNG SỐ
# ==============================================================================

# Cấu hình đường dẫn (BẠN CẦN THAY ĐỔI CÁC ĐƯỜNG DẪN NÀY)
VINDR_ROOT_PATH = "/path/to/your/vinbigdata-cxr" # Thư mục gốc chứa train/, test/, train.csv
OUTPUT_PATH = "/path/to/your/output/RefinedVinDr"

# Định nghĩa bộ nhãn chúng ta muốn giữ lại
DISEASES_TO_KEEP = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Pleural Effusion',
    'Pneumothorax'
]
# Toàn bộ các cột cuối cùng, bao gồm cả 'No Finding'
FINAL_LABEL_SET = ['No Finding'] + DISEASES_TO_KEEP

# Ánh xạ tên bệnh từ VinDr-CXR sang tên chung
VINDR_TO_COMMON_MAP = {
    'Atelectasis': 'Atelectasis',
    'Cardiomegaly': 'Cardiomegaly',
    'Consolidation': 'Consolidation',
    'Pleural effusion': 'Pleural Effusion', # Chuẩn hóa tên
    'Pneumothorax': 'Pneumothorax'
}

# ==============================================================================
# HÀM XỬ LÝ CHÍNH
# ==============================================================================

def preprocess_and_filter_vindr():
    """
    Lọc và sao chép dữ liệu cho tập train của VinDr-CXR.
    """
    print("--- Starting preprocessing for VinDr-CXR train set ---")
    
    # 1. Tạo đường dẫn
    source_csv_path = os.path.join(VINDR_ROOT_PATH, 'train.csv')
    source_image_dir = os.path.join(VINDR_ROOT_PATH, 'train')
    
    if not os.path.exists(source_csv_path) or not os.path.exists(source_image_dir):
        print(f"Error: Source CSV or image directory not found.")
        return

    target_dir = os.path.join(OUTPUT_PATH)
    target_image_dir = os.path.join(target_dir, 'train') # Tạo thư mục con train
    target_csv_path = os.path.join(target_dir, 'refined_train.csv')

    # Tạo thư mục đầu ra
    os.makedirs(target_image_dir, exist_ok=True)

    # 2. Đọc và ánh xạ DataFrame gốc
    print("Reading and mapping original train.csv...")
    df_raw = pd.read_csv(source_csv_path)
    
    # Ánh xạ tên bệnh về tên chung
    df_raw['class_name'] = df_raw['class_name'].replace(VINDR_TO_COMMON_MAP)
    
    # 3. Lọc ra các image_id cần giữ lại
    # Một image_id được giữ lại nếu:
    # - Hoặc nó có ít nhất một bệnh trong DISEASES_TO_KEEP.
    # - Hoặc nó là một ảnh 'No Finding'.
    
    # Lấy ID của các ảnh có bệnh lý liên quan
    ids_with_disease = df_raw[df_raw['class_name'].isin(DISEASES_TO_KEEP)]['image_id'].unique()
    
    # Lấy ID của các ảnh không có phát hiện
    ids_no_finding = df_raw[df_raw['class_name'] == 'No finding']['image_id'].unique()
    
    # Gộp hai danh sách ID lại và loại bỏ trùng lặp
    ids_to_keep = set(ids_with_disease).union(set(ids_no_finding))
    
    print(f"Original number of unique images: {df_raw['image_id'].nunique()}")
    print(f"Number of unique images to keep: {len(ids_to_keep)}")

    # 4. Tạo DataFrame mới đã được định dạng lại (pivoted)
    print("Creating new refined CSV file...")
    # Lọc DataFrame gốc chỉ giữ lại các hàng có image_id cần thiết
    df_filtered_raw = df_raw[df_raw['image_id'].isin(ids_to_keep)].copy()
    
    # Bây giờ, thực hiện pivot trên DataFrame đã được lọc
    # (Đây là logic từ hàm map_vindr_labels đã viết trước đó)
    df_findings = df_filtered_raw[df_filtered_raw['class_name'].isin(DISEASES_TO_KEEP)]
    df_pivot = df_findings.pivot_table(index='image_id', columns='class_name', aggfunc='size', fill_value=0).reset_index()
    
    all_image_ids_to_keep_df = pd.DataFrame(list(ids_to_keep), columns=['image_id'])
    df_wide = pd.merge(all_image_ids_to_keep_df, df_pivot, on='image_id', how='left').fillna(0)
    
    df_wide['No Finding'] = (df_wide[DISEASES_TO_KEEP].sum(axis=1) == 0).astype(int)
    
    # Đảm bảo các cột tồn tại
    for label in FINAL_LABEL_SET:
        if label not in df_wide.columns:
            df_wide[label] = 0
            
    df_final = df_wide[['image_id'] + FINAL_LABEL_SET]

    # 5. Sao chép các file ảnh đã lọc
    print("Copying filtered image files...")
    num_copied = 0
    num_skipped = 0
    for image_id in tqdm(df_final['image_id'], desc="Copying train images"):
        source_image_path = os.path.join(source_image_dir, f"{image_id}.dicom")
        target_image_path = os.path.join(target_image_dir, f"{image_id}.dicom")
        
        try:
            if not os.path.exists(target_image_path):
                shutil.copy2(source_image_path, target_image_path)
            num_copied += 1
        except FileNotFoundError:
            print(f"Warning: Source image not found, skipping: {source_image_path}")
            num_skipped += 1
            # Xóa hàng này khỏi df_final nếu file ảnh không tồn tại
            df_final = df_final[df_final['image_id'] != image_id]
            
    print(f"Finished copying. Copied: {num_copied}, Skipped: {num_skipped}")

    # 6. Lưu DataFrame đã lọc
    print(f"Saving new CSV to {target_csv_path}")
    df_final.to_csv(target_csv_path, index=False)
    
    print(f"--- Preprocessing for VinDr-CXR train set completed! ---\n")

# ==============================================================================
# HÀM MAIN ĐỂ CHẠY
# ==============================================================================

if __name__ == "__main__":
    print("===== Starting VinDr-CXR Dataset Refinement Process =====")
    
    # Chỉ xử lý tập train
    preprocess_and_filter_vindr()
    
    print("===== All tasks completed! =====")
    print(f"Refined dataset is now available at: {OUTPUT_PATH}")