import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from sklearn.metrics import roc_auc_score
import os
from tqdm import tqdm
from omegaconf import OmegaConf

from data_mapping import TRAINING_LABEL_SET
from models import get_model_chexpert_14
from data_loader import get_chexpert_full_label_loaders, get_data_loaders_nih14, get_data_loaders_padchest, get_data_loaders_vindr

FINETUNED_MODEL_PATH = "./Medical/results/mobile_net_14class_jul3_23h59.pth"
# -----------------------------------------------

# Định nghĩa các lớp bệnh để đánh giá
DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 
    'Pleural Effusion', 'Pneumothorax'
]
FINAL_LABEL_SET = ['No Finding'] + DISEASES
NUM_CLASSES = len(FINAL_LABEL_SET)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_pretrained_model(model_path, cfg):
    print(f"Loading fine-tuned weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run the training script first.")
    
    print(f"Found fine-tuned model at {model_path}")
    # Load the pre-trained model architecture
    model = get_model_chexpert_14(cfg)
    # Load the fine-tuned weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    print(f"Loaded fine-tuned model from {model_path}")
    
    print("Fine-tuned model loaded successfully.")
    return model

# def map_chexpert_labels(df_raw):
#     """
#     Ánh xạ nhãn cho CheXpert, xử lý giá trị không chắc chắn (-1).
#     """
#     df_mapped = df_raw[['Path'] + FINAL_LABEL_SET].copy()
#     df_mapped = df_mapped.fillna(0)
#     for col in FINAL_LABEL_SET:
#         if col in df_mapped.columns:
#             # Coi giá trị không chắc chắn là dương tính
#             df_mapped[col] = df_mapped[col].replace(-1.0, 1.0)
#     return df_mapped

# class CheXpertTestDataset(Dataset):
#     def __init__(self, root_path, csv_filename, transform=None):
#         self.root_path = root_path
#         self.transform = transform
        
#         csv_path = os.path.join(root_path, csv_filename)
#         if not os.path.exists(csv_path):
#             raise FileNotFoundError(f"CSV file not found at: {csv_path}")
            
#         print(f">>> Loading and mapping data from {csv_path}...")
#         raw_df = pd.read_csv(csv_path)
#         # self.df = map_chexpert_labels(raw_df)
#         self.df = raw_df
#         print(f"Loaded {len(self.df)} samples for testing.")

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.root_path, self.df.iloc[idx]['image_id'])
#         try:
#             image = Image.open(img_path).convert('RGB')
#         except FileNotFoundError:
#             print(f"Warning: Image not found at {img_path}, skipping.")
#             return None # Sẽ được lọc ra bởi collate_fn

#         labels = self.df.iloc[idx][FINAL_LABEL_SET].values.astype('float')
#         labels = torch.tensor(labels, dtype=torch.float32)
        
#         if self.transform:
#             image = self.transform(image)
            
#         return image, labels

# def collate_fn(batch):
#     """Bỏ qua các mẫu bị lỗi (None) trong batch."""
#     batch = list(filter(lambda x: x is not None, batch))
#     if not batch:
#         return torch.empty(0), torch.empty(0)
#     return torch.utils.data.dataloader.default_collate(batch)

def print_selected_auc_stats(per_class_auc, domain = 'X'):
    # Lấy AUC của các bệnh được chọn
    selected_auc = {f'auc/{disease}': per_class_auc[f'auc/{disease}'] for disease in FINAL_LABEL_SET}
    
    # Tính AUC trung bình
    auc_mean = sum(selected_auc.values()) / len(selected_auc)
    
    # In AUC trung bình
    print(f"Domain {domain} - AUC trung bình: {auc_mean:.4f}")
    
    # In danh sách AUC của các bệnh
    print(f"\nDomain {domain} - Danh sách AUC của các bệnh:")
    for disease, auc in selected_auc.items():
        print(f"{disease}: {auc:.4f}")

def evaluate(model, data_loader, device, label_set = TRAINING_LABEL_SET):
    """
    Đánh giá mô hình, trả về mean AUC, loss trung bình, và một dict chứa AUC của từng lớp.
    """    
    model.eval()  # Chuyển mô hình sang chế độ đánh giá
    
    all_probs = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            if images.numel() == 0: continue
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)  # Logits
            # loss = criterion(outputs, labels)
            # total_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(outputs)  # Chuyển logits thành xác suất
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Tính toán AUC
    auc_scores = {}
    valid_aucs = []
    for i, class_name in enumerate(label_set):
        if len(np.unique(all_labels[:, i])) > 1:
            try:
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                auc_scores[f"auc/{class_name}"] = auc # Tên key phù hợp cho wandb
                valid_aucs.append(auc)
            except ValueError:
                pass
                
    mean_auc = np.mean(valid_aucs) if valid_aucs else 0.0
    avg_loss = total_loss / len(data_loader.dataset)
    
    return mean_auc, avg_loss, auc_scores

def evaluate_model(model, data_loader, device, title="X"):
    """
    Chạy đánh giá mô hình trên tập dữ liệu test và tính AUC.
    """
    print(f"\n>>> Starting evaluation {title}...")
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

def main():
    print(f"Using device: {DEVICE}")

    cfg = OmegaConf.load('Medical/configs/base_config.yaml')
    
    # 1. Tải mô hình
    model = get_pretrained_model(model_path=FINETUNED_MODEL_PATH, cfg=cfg)
    
    # 2. Chuẩn bị DataLoader
    # Định nghĩa các phép biến đổi ảnh cho tập test
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n>>> Loading datasets...")
    # 3. Chạy đánh giá
    
    # Base 
    # _, chexpert_test_loader =  get_chexpert_full_label_loaders(cfg)
    # mean_valid_auc, epoch_valid_loss, per_class_auc = evaluate(model, chexpert_test_loader, DEVICE, criterion)
    # print_selected_auc_stats(per_class_auc)
    
    # VinDr CXR 
    # vindr_test_loader =  get_data_loaders_vindr(cfg)
    # mean_valid_auc, epoch_valid_loss, per_class_auc = evaluate(model, vindr_test_loader, DEVICE, FINAL_LABEL_SET)
    # print_selected_auc_stats(per_class_auc, 'VinDr')
    # # evaluate_model(model, vindr_test_loader, DEVICE, "VinData")
    
    # NIH 14 dataset
    # nih14_test_loader = get_data_loaders_nih14(cfg)
    # mean_valid_auc, epoch_valid_loss, per_class_auc = evaluate(model, nih14_test_loader, DEVICE, FINAL_LABEL_SET)
    # print_selected_auc_stats(per_class_auc, "nih-14")
    # # evaluate_model(model, nih14_test_loader, DEVICE, "nih_14")
    
    # PadChest dataset
    padchest_test_loader = get_data_loaders_padchest(cfg)
    mean_valid_auc, epoch_valid_loss, per_class_auc = evaluate(model, padchest_test_loader, DEVICE, FINAL_LABEL_SET)
    print_selected_auc_stats(per_class_auc, "padchest")
    # evaluate_model(model, padchest_test_loader, DEVICE, "padchest")

    # 3. Tạo luồng dữ liệu liên tục
    # eval_transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    
    # Chuỗi các domain thay đổi theo từng batch
    # Ví dụ: 100 batch, luân phiên
    # domain_sequence = ['vindr', 'nih14', 'padchest'] * 33 + ['vindr'] 
    
    # continual_loader = ContinualDomainLoader(
    #     cfg, 
    #     domain_sequence=domain_sequence, 
    #     batch_size=cfg.TRAINING.BATCH_SIZE, 
    #     transform=eval_transform
    # )
    # evaluate_model(model, continual_loader, DEVICE, "honhop")

if __name__ == "__main__":
    main()