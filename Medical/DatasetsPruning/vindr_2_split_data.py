import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SOURCE_DATA_DIR = '/Users/admin/Working/Data/vinbigdata-chest-xray'
DEST_DATA_DIR = '/Users/admin/Working/Data/vinbigdata-chest-xray-30-percent'
SAMPLE_FRACTION = 0.3
RANDOM_STATE = 42

# Define condition columns for stratification (excluding 'image_id' and the last column)
CONDITION_COLUMNS = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Pleural Effusion', 'Pneumothorax']

DEST_TRAIN_IMAGE_DIR = os.path.join(DEST_DATA_DIR, 'train')
os.makedirs(DEST_TRAIN_IMAGE_DIR, exist_ok=True)
print(f"New directory created at: {DEST_DATA_DIR}")

SOURCE_TRAIN_CSV_PATH = os.path.join(SOURCE_DATA_DIR, 'validate.csv')
SOURCE_TRAIN_IMAGE_DIR = os.path.join(SOURCE_DATA_DIR, 'images')

print("Loading original train.csv...")
df_original = pd.read_csv(SOURCE_TRAIN_CSV_PATH)
print(f"Original data rows: {len(df_original)}")

# Create a 'condition_present' column for stratification (1 if any condition is 1, 0 if only 'No Finding' is 1)
df_original['condition_present'] = df_original[CONDITION_COLUMNS].apply(lambda row: 1 if row.sum() > 1 or (row['No Finding'] == 0 and row.sum() == 1) else 0, axis=1)

print(f"Performing stratified sampling for {SAMPLE_FRACTION*100}% of data...")
df_sampled_rows, _ = train_test_split(
    df_original,
    train_size=SAMPLE_FRACTION,
    stratify=df_original['condition_present'],
    random_state=RANDOM_STATE
)

# Get unique image_ids from sampled rows
sampled_image_ids = df_sampled_rows['image_id'].unique()
print(f"Number of unique images selected: {len(sampled_image_ids)}")

# Filter to include all annotations for sampled image_ids
df_final_csv = df_original[df_original['image_id'].isin(sampled_image_ids)]

# Save new CSV maintaining original structure
DEST_TRAIN_CSV_PATH = os.path.join(DEST_DATA_DIR, 'train.csv')
df_final_csv.to_csv(DEST_TRAIN_CSV_PATH, index=False)
print(f"New train.csv created at: {DEST_TRAIN_CSV_PATH}")
print(f"New train.csv rows: {len(df_final_csv)}")

print(f"\nCopying {len(sampled_image_ids)} DICOM files...")
for image_id in tqdm(sampled_image_ids, desc="Copying images"):
    source_path = os.path.join(SOURCE_TRAIN_IMAGE_DIR, f"{image_id}.dicom")
    dest_path = os.path.join(DEST_TRAIN_IMAGE_DIR, f"{image_id}.dicom")
    if os.path.exists(source_path):
        shutil.copy(source_path, dest_path)
    else:
        print(f"Warning: Source file not found: {source_path}")

print("\n--- DISTRIBUTION COMPARISON ---")
print("Original data distribution (condition_present):")
print(df_original['condition_present'].value_counts(normalize=True).round(4))

print("\nSampled data distribution (condition_present):")
print(df_final_csv['condition_present'].value_counts(normalize=True).round(4))

print("\nProcess completed successfully!")