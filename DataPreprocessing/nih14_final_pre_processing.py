import pandas as pd
import os
import shutil
from tqdm import tqdm
import glob
import argparse

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Define the original labels in NIH14 that we want to retain
# Note: 'Effusion' in NIH14 will be mapped to 'Pleural Effusion'
DISEASES_TO_KEEP_NIH14 = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Effusion',  # Original name in NIH14
    'Pneumothorax'
]

# Define the final standardized label set to align with other datasets
FINAL_LABEL_SET = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Pleural Effusion',
    'Pneumothorax'
]


def step_1_scan_image_paths(root_path: str) -> dict:
    """
    Step 1: Scan all 'images*' subfolders to index image paths.
    """
    print("--- Step 1: Scanning and indexing image files ---")
    all_image_paths = {}
    image_folders = glob.glob(os.path.join(root_path, 'images*'))
    if not image_folders:
        raise FileNotFoundError(f"No 'images*' folder found in: {root_path}")

    for folder in tqdm(image_folders, desc="Scanning image folders"):
        final_folder_path = os.path.join(folder, 'images')
        if not os.path.isdir(final_folder_path):
            print(f"Warning: Skipping invalid folder: {final_folder_path}")
            continue
            
        for img_file in os.listdir(final_folder_path):
            if img_file.endswith('.png'):
                all_image_paths[img_file] = os.path.join(final_folder_path, img_file)
    
    print(f" -> Total images found: {len(all_image_paths)}")
    print("--- Step 1: Completed ---\n")
    return all_image_paths


def step_2_filter_and_standardize_csv(source_csv_path: str) -> pd.DataFrame:
    """
    Step 2: Read CSV file, filter rows with target diseases, and standardize labels.
    """
    print("--- Step 2: Reading, filtering, and standardizing CSV file ---")
    
    if not os.path.exists(source_csv_path):
        raise FileNotFoundError(f"Source CSV file not found: {source_csv_path}")

    df_raw = pd.read_csv(source_csv_path)
    df_raw.rename(columns={'Finding Labels': 'labels', 'Image Index': 'image_id'}, inplace=True)
    
    filter_condition = df_raw['labels'].str.contains('|'.join(DISEASES_TO_KEEP_NIH14))
    df_filtered = df_raw[filter_condition].copy()
    
    print(f" -> Original record count: {len(df_raw)}")
    print(f" -> Records after filtering: {len(df_filtered)}")

    print(" -> Creating one-hot encoded label columns...")
    for disease_final, disease_original in zip(FINAL_LABEL_SET, DISEASES_TO_KEEP_NIH14):
        df_filtered[disease_final] = df_filtered['labels'].apply(lambda x: 1 if disease_original in x else 0)

    columns_to_save = ['image_id'] + FINAL_LABEL_SET
    df_final = df_filtered[columns_to_save]

    print("--- Step 2: Completed ---\n")
    return df_final


def step_3_copy_files_and_save_csv(df_final: pd.DataFrame, all_image_paths: dict, output_dir: str):
    """
    Step 3: Copy filtered image files and save the final DataFrame.
    """
    print("--- Step 3: Copying images and saving final CSV file ---")
    
    target_image_dir = os.path.join(output_dir, 'images')
    target_csv_path = os.path.join(output_dir, 'validate.csv')
    os.makedirs(target_image_dir, exist_ok=True)
    
    num_copied = 0
    images_not_found = []
    
    for image_id in tqdm(df_final['image_id'], desc="Copying filtered images"):
        if image_id in all_image_paths:
            source_path = all_image_paths[image_id]
            target_path = os.path.join(target_image_dir, image_id)
            if not os.path.exists(target_path):
                shutil.copy2(source_path, target_path)
            num_copied += 1
        else:
            images_not_found.append(image_id)
    
    print(f" -> Copying completed. Total copied: {num_copied} images.")
    
    if images_not_found:
        print(f"Warning: {len(images_not_found)} images not found and will be removed from CSV.")
        df_final = df_final[~df_final['image_id'].isin(images_not_found)]
        
    print(f" -> Saving new CSV to: {target_csv_path}")
    df_final.to_csv(target_csv_path, index=False)
    
    print("--- Step 3: Completed ---\n")


def main(nih_root_path: str, output_path: str):
    """
    Main function to coordinate the NIH14 data processing pipeline.
    """
    print("Starting NIH14 dataset processing pipeline")
    print(f"Source directory: {nih_root_path}")
    print(f"Output directory: {output_path}\n")

    try:
        all_image_paths = step_1_scan_image_paths(nih_root_path)
        source_csv_path = os.path.join(nih_root_path, 'Data_Entry_2017.csv')
        df_processed = step_2_filter_and_standardize_csv(source_csv_path)
        step_3_copy_files_and_save_csv(df_processed, all_image_paths, output_path)
        print(">>> NIH14 DATA PROCESSING COMPLETED SUCCESSFULLY! <<<")

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        print("Pipeline aborted.")


if __name__ == "__main__":
    nih_path = '/Users/admin/Working/Data/nih-14-origin'    
    output_path = '/Users/admin/Working/Data/nih_14_structured'    
    main(nih_path, output_path)