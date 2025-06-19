import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from PIL import Image
import os
from tqdm import tqdm
from omegaconf import OmegaConf

from models import get_model

# ==============================================================================
# PHẦN 1: CẤU HÌNH VÀ ĐỊNH NGHĨA
# ==============================================================================

# ----- BẠN CẦN THAY ĐỔI CÁC ĐƯỜNG DẪN NÀY -----
CHEXPERT_PATH = "./datasets/CheXpert-v1.0-small" # Đường dẫn đến bộ dữ liệu gốc
FINETUNED_MODEL_PATH = "./results/finetuned_model.pth"
TEST_CSV_FILENAME = "valid.csv" # Dùng tập valid gốc để test
# -----------------------------------------------

# Định nghĩa các lớp bệnh để đánh giá
DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 
    'Pleural Effusion', 'Pneumothorax'
]
FINAL_LABEL_SET = ['No Finding'] + DISEASES
NUM_CLASSES = len(FINAL_LABEL_SET)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# PHẦN 2: CHUẨN BỊ MÔ HÌNH
# ==============================================================================

def get_pretrained_model(num_classes, model_path):
    print(f"Loading fine-tuned weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run the training script first.")
    
    print(f"Found fine-tuned model at {model_path}")
    # Step 1: Load the pre-trained model architecture
    cfg = OmegaConf.load('configs/base_config.yaml')
    device = torch.device(cfg.TRAINING.DEVICE if torch.cuda.is_available() else "cpu")
    model = get_model(cfg)
    # Load the fine-tuned weights
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    print(f"Loaded fine-tuned model from {model_path}")
    
    print("Fine-tuned model loaded successfully.")
    return model

# ==============================================================================
# PHẦN 3: CHUẨN BỊ DỮ LIỆU
# ==============================================================================

def map_chexpert_labels(df_raw):
    """
    Ánh xạ nhãn cho CheXpert, xử lý giá trị không chắc chắn (-1).
    """
    df_mapped = df_raw[['Path'] + FINAL_LABEL_SET].copy()
    df_mapped = df_mapped.fillna(0)
    for col in FINAL_LABEL_SET:
        if col in df_mapped.columns:
            # Coi giá trị không chắc chắn là dương tính
            df_mapped[col] = df_mapped[col].replace(-1.0, 1.0)
    return df_mapped

class CheXpertTestDataset(Dataset):
    def __init__(self, root_path, csv_filename, transform=None):
        self.root_path = root_path
        self.transform = transform
        
        csv_path = os.path.join(root_path, csv_filename)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")
            
        print(f">>> Loading and mapping data from {csv_path}...")
        raw_df = pd.read_csv(csv_path)
        self.df = map_chexpert_labels(raw_df)
        print(f"Loaded {len(self.df)} samples for testing.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_path, self.df.iloc[idx]['Path'])
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image not found at {img_path}, skipping.")
            return None # Sẽ được lọc ra bởi collate_fn

        labels = self.df.iloc[idx][FINAL_LABEL_SET].values.astype('float')
        labels = torch.tensor(labels, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels

def collate_fn(batch):
    """Bỏ qua các mẫu bị lỗi (None) trong batch."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)


# ==============================================================================
# PHẦN 4: HÀM ĐÁNH GIÁ
# ==============================================================================

def evaluate_model(model, data_loader, device):
    """
    Chạy đánh giá mô hình trên tập dữ liệu test và tính AUC.
    """
    print("\n>>> Starting evaluation...")
    model.to(device)
    model.eval() # Chuyển mô hình sang chế độ đánh giá

    all_probs = []
    all_labels = []
    
    with torch.no_grad(): # Không cần tính gradient khi đánh giá
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            if images.numel() == 0: continue
            
            images = images.to(device)
            
            outputs = model(images) # Lấy logits từ mô hình
            probs = torch.sigmoid(outputs) # Chuyển logits thành xác suất
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            
    # Gộp kết quả từ tất cả các batch
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Tính toán AUC cho từng lớp và AUC trung bình
    print("\n--- AUC Scores per Class ---")
    auc_scores = []
    for i, class_name in enumerate(FINAL_LABEL_SET):
        # Chỉ tính AUC nếu có cả nhãn dương và âm
        if len(np.unique(all_labels[:, i])) > 1:
            try:
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                auc_scores.append(auc)
                print(f"{class_name:<20}: {auc:.4f}")
            except ValueError:
                print(f"Warning: AUC undefined for class {class_name}, skipping.")
        else:
            print(f"Warning: Only one class present for {class_name}, skipping AUC calculation.")

    mean_auc = np.mean(auc_scores) if auc_scores else 0.0
    print("---------------------------------")
    print(f"Mean AUC: {mean_auc:.4f}")
    print("---------------------------------")
    
    return mean_auc


# ==============================================================================
# PHẦN 5: HÀM MAIN ĐỂ CHẠY
# ==============================================================================

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Tải mô hình
    model = get_pretrained_model(num_classes=NUM_CLASSES,
        model_path=FINETUNED_MODEL_PATH)
    
    # 2. Chuẩn bị DataLoader
    # Định nghĩa các phép biến đổi ảnh cho tập test
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = CheXpertTestDataset(
        root_path=CHEXPERT_PATH,
        csv_filename=TEST_CSV_FILENAME,
        transform=eval_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32, # Có thể tăng batch size khi đánh giá
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # 3. Chạy đánh giá
    evaluate_model(model, test_loader, DEVICE)

if __name__ == "__main__":
    main()