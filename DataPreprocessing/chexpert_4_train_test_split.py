import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def split_chexpert_csv(train_ratio=0.8, random_state=42):
    """
    Split CheXpert train.csv into train and validation sets with 80-20 ratio,
    maintaining label distribution for specified columns.
    
    Args:
        cfg: Configuration object with DATA.CHEXPERT_PATH and DATA.CHEXPERT_TRAIN_CSV attributes
        train_ratio (float): Proportion of data for training set (default: 0.8)
        random_state (int): Random seed for reproducibility
    """
    # Define the label columns
    TRAINING_LABEL_SET = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Pleural Effusion', 'Pneumothorax']
    
    # Read the input CSV
    CHEXPERT_PATH = "/home/ngoto/Working/Data/CheXpert-v1.0-small"
    # CHEXPERT_PATH_ROOT_PATH: "/Users/admin/Working/Data"
    # CHEXPERT_PATH: "/Users/admin/Working/Data/CheXpert-v1.0-small"
    CHEXPERT_TRAIN_CSV = "train_final_reordered.csv"
    csv_path = os.path.join(CHEXPERT_PATH, CHEXPERT_TRAIN_CSV)
    df = pd.read_csv(csv_path)
    
    # Verify that all required columns exist
    missing_cols = [col for col in TRAINING_LABEL_SET + ['image_id'] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")
    
    # Handle CheXpert uncertain labels (-1) by mapping to 0 or 1 (e.g., treat -1 as 0)
    labels = df[TRAINING_LABEL_SET].fillna(0).replace(-1, 0).astype(int)
    
    # Create a stratification key based on label combinations
    stratify_key = labels.apply(lambda x: ''.join(x.astype(str)), axis=1)
    
    # Perform stratified split
    train_df, valid_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=stratify_key,
        random_state=random_state
    )
    
    # Save the resulting CSVs
    train_csv_path = os.path.join(CHEXPERT_PATH, 'train_split.csv')
    valid_csv_path = os.path.join(CHEXPERT_PATH, 'valid_split.csv')
    
    train_df.to_csv(train_csv_path, index=False)
    valid_df.to_csv(valid_csv_path, index=False)
    
    print(f"Saved training set with {len(train_df)} samples to {train_csv_path}")
    print(f"Saved validation set with {len(valid_df)} samples to {valid_csv_path}")
    
    # Verify label distribution
    print("\nLabel distribution in training set (mean):")
    print(train_df[TRAINING_LABEL_SET].mean())
    print("\nLabel distribution in validation set (mean):")
    print(valid_df[TRAINING_LABEL_SET].mean())
    
    return train_df, valid_df


split_chexpert_csv()