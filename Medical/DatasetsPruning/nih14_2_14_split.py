import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob

SOURCE_DATA_DIR = '/Users/admin/Working/Data/nih-14-pruning'
DEST_DATA_DIR = '/Users/admin/Working/Data/cxr-14-30_percent'

SAMPLE_FRACTION = 0.3
RANDOM_STATE = 42

# Các cột nhãn trong file CSV
LABEL_COLUMNS = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Pleural Effusion', 'Pneumothorax']

DEST_TEST_IMAGE_DIR = os.path.join(DEST_DATA_DIR, 'images')
os.makedirs(DEST_TEST_IMAGE_DIR, exist_ok=True)
print(f"New directory created at: {DEST_DATA_DIR}")

SOURCE_CSV_PATH = os.path.join(SOURCE_DATA_DIR, 'Data_Entry_2017.csv')

print("Loading original Data_Entry_2017.csv...")
df_original = pd.read_csv(SOURCE_CSV_PATH)
print(f"Original data rows: {len(df_original)}")

# Đổi tên cột 'Image Index' thành 'image_id' cho nhất quán
df_original.rename(columns={'Image Index': 'image_id'}, inplace=True)

# Kiểm tra xem các cột nhãn có tồn tại trong file CSV không
missing_labels = [col for col in LABEL_COLUMNS if col not in df_original.columns]
if missing_labels:
    raise ValueError(f"Missing label columns in CSV: {missing_labels}")

# Tạo cột stratify_key để phân tầng dựa trên tổ hợp nhãn
df_original['stratify_key'] = df_original[LABEL_COLUMNS].apply(lambda x: ''.join(x.astype(str)), axis=1)

print("Filtering out rare label combinations (fewer than 2 instances)...")
# Đếm số lần xuất hiện của mỗi stratify_key
key_counts = df_original['stratify_key'].value_counts()
# Giữ các stratify_key có ít nhất 2 mẫu
valid_keys = key_counts[key_counts >= 2].index
df_filtered = df_original[df_original['stratify_key'].isin(valid_keys)]
print(f"Rows after filtering rare combinations: {len(df_filtered)} (removed {len(df_original) - len(df_filtered)} rows)")

print(f"Performing stratified sampling for {SAMPLE_FRACTION*100}% of filtered data...")
# Phân tầng dựa trên stratify_key để duy trì phân phối nhãn
_, df_sampled = train_test_split(
    df_filtered,
    test_size=SAMPLE_FRACTION,  # Lấy 10% làm tập test
    stratify=df_filtered['stratify_key'],  # Phân tầng theo tổ hợp nhãn
    random_state=RANDOM_STATE
)

# Xóa cột stratify_key khỏi df_sampled để giữ cấu trúc gốc
df_sampled = df_sampled.drop(columns=['stratify_key'])

sampled_image_ids = df_sampled['image_id'].unique()
print(f"Number of unique images selected: {len(sampled_image_ids)}")

DEST_TEST_CSV_PATH = os.path.join(DEST_DATA_DIR, 'test.csv')
df_sampled.to_csv(DEST_TEST_CSV_PATH, index=False)
print(f"New test.csv created at: {DEST_TEST_CSV_PATH}")
print(f"New test.csv rows: {len(df_sampled)}")

print(f"\nFinding paths for {len(sampled_image_ids)} PNG files...")
# Tạo một từ điển để lưu đường dẫn của tất cả các ảnh để tìm kiếm nhanh hơn
all_image_paths = {}
image_folders = glob.glob(os.path.join(SOURCE_DATA_DIR, 'images*'))
for folder in tqdm(image_folders, desc="Scanning image folders"):
    for img_file in os.listdir(folder):
        if img_file.endswith('.png'):
            all_image_paths[img_file] = os.path.join(folder, img_file)

print(f"Found {len(all_image_paths)} total images. Now copying sampled files...")

copied_count = 0
not_found_count = 0
for image_id in tqdm(sampled_image_ids, desc="Copying images"):
    if image_id in all_image_paths:
        source_path = all_image_paths[image_id]
        dest_path = os.path.join(DEST_TEST_IMAGE_DIR, image_id)
        shutil.copy(source_path, dest_path)
        copied_count += 1
    else:
        # print(f"Warning: Source file not found for image_id: {image_id}")
        not_found_count += 1

print(f"Copying completed. Copied: {copied_count}, Not Found: {not_found_count}")

# COMPARE DISTRIBUTIONS ---
print("\n--- DISTRIBUTION COMPARISON ---")
print("Original data distribution (per label):")
original_dist = df_original[LABEL_COLUMNS].mean().round(4)  # Tỷ lệ trung bình của mỗi nhãn
print(original_dist)

print("\nSampled data distribution (per label):")
sampled_dist = df_sampled[LABEL_COLUMNS].mean().round(4)  # Tỷ lệ trung bình của mỗi nhãn
print(sampled_dist)

print("\nProcess completed successfully!")