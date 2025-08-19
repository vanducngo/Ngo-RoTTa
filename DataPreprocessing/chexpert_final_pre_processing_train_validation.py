import pandas as pd
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

TARGET_DISEASES = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Pleural Effusion',
    'Pneumothorax',
]

METADATA_COLS_TO_DROP = ['Sex', 'Age', 'Frontal/Lateral', 'AP/PA']
TRAIN_RATIO = 0.8
RANDOM_STATE = 42

def step_1_clean_and_restructure(df: pd.DataFrame) -> pd.DataFrame:
    print("--- Step 1: Start cleaning and restructuring the data ---")

    if 'Path' in df.columns:
        df = df.rename(columns={'Path': 'image_id'})
    
    cols_to_drop = [col for col in METADATA_COLS_TO_DROP if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    finding_cols = [col for col in df.columns if col != 'image_id']
    
    for col in finding_cols:
        df[col] = df[col].fillna(0.0)
        df[col] = df[col].replace(-1.0, 0.0)
    print(" -> Handled uncertain labels (-1.0) and NaNs by converting them to 0.0")
    
    print("--- Step 1: Completed ---\n")
    return df

def step_2_reduce_columns(df: pd.DataFrame, target_diseases: list) -> pd.DataFrame:
    print("--- Step 2: Start reducing the number of columns ---")    
    columns_to_keep = ['image_id'] + target_diseases
    
    missing_cols = [col for col in columns_to_keep if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Lỗi: Các cột mục tiêu sau không có trong dữ liệu: {missing_cols}")
        
    df_reduced = df[columns_to_keep]
    print(f" -> Kept {len(df_reduced.columns)} columns: ['image_id'] and {len(target_diseases)} target diseases.")
    print("--- Step 2: Completed ---\n")
    
    return df_reduced

def step_3_filter_positive_cases(df: pd.DataFrame, diseases: list) -> pd.DataFrame:
    print("--- Step 3: Start filtering positive disease cases ---")
    initial_rows = len(df)
    df_filtered = df[df[diseases].eq(1.0).any(axis=1)].copy()
    
    final_rows = len(df_filtered)
    print(f" -> Filtered data. Kept {final_rows} / {initial_rows} rows (removed samples without any disease).")
    print("--- Step 3: Completed ---\n")
    
    return df_filtered

def step_4_split_data(df: pd.DataFrame, target_diseases: list, output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("--- Step 4: Start splitting the data into train and validation sets ---")
    # Reordering is implicitly done by step 2, but this ensures it
    df = df[['image_id'] + target_diseases]

    stratify_key = df[target_diseases].apply(lambda x: ''.join(x.astype(str)), axis=1)
    
    train_df, valid_df = train_test_split(
        df,
        train_size=TRAIN_RATIO,
        stratify=stratify_key,
        random_state=RANDOM_STATE
    )
    
    train_output_path = os.path.join(output_dir, 'train_final.csv')
    valid_output_path = os.path.join(output_dir, 'valid_final.csv')
    
    train_df.to_csv(train_output_path, index=False)
    valid_df.to_csv(valid_output_path, index=False)
    
    print(f" -> Training set ({len(train_df)} samples) has been saved to: {train_output_path}")
    print(f" -> Validation set ({len(valid_df)} samples) has been saved to: {valid_output_path}")
    print("--- Step 4: Completed ---\n")
    
    return train_df, valid_df

def step_5_visualize_distribution(train_df: pd.DataFrame, valid_df: pd.DataFrame, target_diseases: list, output_path: str):
    print("--- Step 5: Start generating detailed label distribution plots ---")
    
    train_counts = train_df[target_diseases].sum()
    valid_counts = valid_df[target_diseases].sum()
    train_total = len(train_df)
    valid_total = len(valid_df)
    train_percentages = (train_counts / train_total) * 100
    valid_percentages = (valid_counts / valid_total) * 100

    x = np.arange(len(target_diseases))
    width = 0.35
    fig, ax = plt.subplots(figsize=(15, 9))
    
    rects1 = ax.bar(x - width/2, train_counts, width, label=f'Train Set ({train_total} mẫu)')
    rects2 = ax.bar(x + width/2, valid_counts, width, label=f'Validation Set ({valid_total} mẫu)')
    
    ax.set_ylabel('Số lượng ca dương tính (Count)')
    ax.set_title('So sánh phân bổ nhãn bệnh (Số lượng và Tỷ lệ %)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(target_diseases, rotation=20, ha='right', fontsize=12)
    ax.legend(fontsize=12)
    
    train_labels = [f'{c}\n({p:.1f}%)' for c, p in zip(train_counts, train_percentages)]
    valid_labels = [f'{c}\n({p:.1f}%)' for c, p in zip(valid_counts, valid_percentages)]

    ax.bar_label(rects1, labels=train_labels, padding=5, fontsize=10)
    ax.bar_label(rects2, labels=valid_labels, padding=5, fontsize=10)

    ax.set_ylim(0, max(train_counts.max(), valid_counts.max()) * 1.2)
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    
    print(f" -> Detailed plot has been saved to: {output_path}")
    print("--- Step 5: Completed ---")

def main(input_csv_path: str):
    print(f"Starting processing pipeline for file: {input_csv_path}\n")    
    if not os.path.exists(input_csv_path):
        print(f"ERROR: Input file not found at '{input_csv_path}'")
        return
        
    output_dir = os.path.dirname(input_csv_path)
    
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"ERROR: Failed to read CSV file. Details: {e}")
        return
    
    df_step1 = step_1_clean_and_restructure(df)
    df_step2 = step_2_reduce_columns(df_step1, TARGET_DISEASES)
    df_step3 = step_3_filter_positive_cases(df_step2, TARGET_DISEASES)
    
    train_df, valid_df = step_4_split_data(df_step3, TARGET_DISEASES, output_dir)
    chart_output_path = os.path.join(output_dir, 'label_distribution_final.png')
    step_5_visualize_distribution(train_df, valid_df, TARGET_DISEASES, chart_output_path)
    
    print("\n>>> PROCESSING PIPELINE COMPLETED! <<<")

if __name__ == "__main__":
    input_file = '/home/ngoto/Working/Data/CheXpert-v1.0-small/train_origin.csv'    
    main(input_file)