import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob

SOURCE_CSV_PATH = '/home/ngoto/Working/Data/MixData/nih_14_structured/validate_reordered.csv'
DEST_TEST_CSV_PATH = '/home/ngoto/Working/Data/MixData/nih_14_structured/validate_reordered_subset.csv'

SAMPLE_FRACTION = 0.2
RANDOM_STATE = 10

# Các cột nhãn trong file CSV
LABEL_COLUMNS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Pleural Effusion', 'Pneumothorax']
    
print("Loading original Data_Entry_2017.csv...")
df_original = pd.read_csv(SOURCE_CSV_PATH)
print(f"Original data rows: {len(df_original)}")

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

df_sampled.to_csv(DEST_TEST_CSV_PATH, index=False)
print(f"New test.csv created at: {DEST_TEST_CSV_PATH}")
print(f"New test.csv rows: {len(df_sampled)}")

# COMPARE DISTRIBUTIONS ---
print("\n--- DISTRIBUTION COMPARISON ---")
print("Original data distribution (per label):")
original_dist = df_original[LABEL_COLUMNS].mean().round(4)  # Tỷ lệ trung bình của mỗi nhãn
print(original_dist)

print("\nSampled data distribution (per label):")
sampled_dist = df_sampled[LABEL_COLUMNS].mean().round(4)  # Tỷ lệ trung bình của mỗi nhãn
print(sampled_dist)

print("\nProcess completed successfully!")