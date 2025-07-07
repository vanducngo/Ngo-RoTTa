import pandas as pd
import os
import shutil
from tqdm import tqdm

import glob

NIH14_ROOT_PATH = "/Users/admin/Working/Data/nih-14-origin" # Thư mục gốc chứa images/ và Data_Entry_2017.csv
OUTPUT_PATH = "/Users/admin/Working/Data/nih-14-pruning-5-class"

# Định nghĩa bộ nhãn chúng ta muốn giữ lại
DISEASES_TO_KEEP = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Effusion', # Tên gốc trong NIH14
    'Pneumothorax'
]
# FINAL_LABEL_SET_MAPPED = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Pleural Effusion', 'Pneumothorax']
FINAL_LABEL_SET_MAPPED = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Pleural Effusion', 'Pneumothorax']

def preprocess_and_filter_nih14():
    """
    Lọc và sao chép dữ liệu cho bộ dữ liệu NIH ChestX-ray14.
    """
    print("--- Starting preprocessing for NIH ChestX-ray14 set ---")
    
    source_csv_path = os.path.join(NIH14_ROOT_PATH, 'Data_Entry_2017.csv')
    
    # Tạo một từ điển để lưu đường dẫn của tất cả các ảnh để tìm kiếm nhanh hơn
    all_image_paths = {}
    image_folders = glob.glob(os.path.join(NIH14_ROOT_PATH, 'images*'))
    for folder in tqdm(image_folders, desc="Scanning image folders"):
        finalFolder = f'{folder}/images'
        for img_file in os.listdir(finalFolder):
            if img_file.endswith('.png'):
                all_image_paths[img_file] = os.path.join(finalFolder, img_file)

    print(f"Found {len(all_image_paths)} total images. Now copying sampled files...")

    target_dir = os.path.join(OUTPUT_PATH)
    target_image_dir = os.path.join(target_dir, 'images') # Tạo thư mục con images
    target_csv_path = os.path.join(target_dir, 'validate.csv')

    # Tạo thư mục đầu ra
    os.makedirs(target_image_dir, exist_ok=True)

    # 2. Đọc và lọc DataFrame gốc
    print("Reading and filtering original Data_Entry_2017.csv...")
    df_raw = pd.read_csv(source_csv_path)
    
    # Đổi tên cột cho tiện
    df_raw.rename(columns={'Finding Labels': 'labels', 'Image Index': 'image_id'}, inplace=True)
    
    # Điều kiện lọc: giữ lại một hàng nếu nhãn của nó chứa 'No Finding' hoặc bất kỳ bệnh nào trong DISEASES_TO_KEEP
    # df_raw['labels'].str.contains('A|B') là cách tìm các chuỗi chứa A hoặc B
    filter_condition = df_raw['labels'].str.contains('No Finding|' + '|'.join(DISEASES_TO_KEEP))
    df_filtered = df_raw[filter_condition].copy()
    
    print(f"Original number of records: {len(df_raw)}")
    print(f"Number of records after filtering: {len(df_filtered)}")

    # 3. Tạo các cột nhãn nhị phân và chuẩn hóa tên
    print("Creating one-hot encoded labels...")
    for disease in FINAL_LABEL_SET_MAPPED:
        # Xử lý trường hợp đặc biệt của Pleural Effusion
        if disease == 'Pleural Effusion':
            df_filtered[disease] = df_filtered['labels'].apply(lambda x: 1 if 'Effusion' in x else 0)
        else:
            df_filtered[disease] = df_filtered['labels'].apply(lambda x: 1 if disease in x else 0)

    # Chỉ giữ lại các cột cần thiết cho file CSV mới
    columns_to_save = ['image_id'] + FINAL_LABEL_SET_MAPPED
    df_final = df_filtered[columns_to_save]

    # 4. Sao chép các file ảnh đã lọc
    print("Copying filtered image files...")
    num_copied = 0
    num_skipped = 0
    for image_id in tqdm(df_final['image_id'], desc="Copying images"):
        # source_image_path = os.path.join(source_image_dir, image_id)
        source_image_path = all_image_paths[image_id]
        target_image_path = os.path.join(target_image_dir, image_id)
        
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

    # 5. Lưu DataFrame đã lọc
    print(f"Saving new CSV to {target_csv_path}")
    df_final.to_csv(target_csv_path, index=False)
    
    print(f"--- Preprocessing for NIH ChestX-ray14 set completed! ---\n")

if __name__ == "__main__":
    preprocess_and_filter_nih14()