import os
import torch
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm

from constants import COMMON_FINAL_LABEL_SET, TRAINING_LABEL_SET
from models import get_model_chexpert_14

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, data_loader, criterion = None):
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
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)  # Logits

            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(outputs)  # Chuyển logits thành xác suất
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Tính toán AUC
    auc_scores = {}
    valid_aucs = []
    for i, class_name in enumerate(TRAINING_LABEL_SET):
        if class_name in COMMON_FINAL_LABEL_SET:
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

# def evaluate_model(model, data_loader, device, criterion = None):
#     """
#     Đánh giá mô hình, trả về mean AUC, loss trung bình, và một dict chứa AUC của từng lớp.
#     """    
#     model.eval()  # Chuyển mô hình sang chế độ đánh giá
    
#     all_probs = []
#     all_labels = []
#     total_loss = 0.0
    
#     with torch.no_grad():
#         for images, labels in tqdm(data_loader, desc="Evaluating"):
#             if images.numel() == 0: continue
#             images, labels = images.to(device), labels.to(device)
            
#             outputs = model(images)  # Logits

#             loss = 0
#             total_loss = 0
#             if criterion is not None:
#                 loss = criterion(outputs, labels)
#                 total_loss += loss.item() * images.size(0)
            
#             probs = torch.sigmoid(outputs)  # Chuyển logits thành xác suất
#             all_probs.append(probs.cpu().numpy())
#             all_labels.append(labels.cpu().numpy())
            
#     all_probs = np.concatenate(all_probs, axis=0)
#     all_labels = np.concatenate(all_labels, axis=0)
    
#     # Tính toán AUC
#     auc_scores = {}
#     valid_aucs = []
#     for i, class_name in enumerate(COMMON_FINAL_LABEL_SET):
#         if len(np.unique(all_labels[:, i])) > 1:
#             try:
#                 auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
#                 auc_scores[f"auc/{class_name}"] = auc # Tên key phù hợp cho wandb
#                 valid_aucs.append(auc)
#             except ValueError:
#                 pass
                
#     mean_auc = np.mean(valid_aucs) if valid_aucs else 0.0
#     avg_loss = total_loss / len(data_loader.dataset)
    
#     print(f'mean_auc: {mean_auc}')
#     print(f'all auc: {auc_scores}')
#     return mean_auc, avg_loss, auc_scores

def calculate_auc(all_preds, all_labels):
    """
    Tính toán AUC trung bình trên tất cả các lớp.
    """
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    num_classes = all_labels.shape[1]
    auc_scores = []
    
    for i in range(num_classes):
        # Chỉ tính AUC nếu có cả nhãn dương và âm
        if len(np.unique(all_labels[:, i])) > 1:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            auc_scores.append(auc)
            
    mean_auc = np.mean(auc_scores)
    return mean_auc

def print_selected_auc_stats(per_class_auc, domain='X'):
    # Lấy AUC của các bệnh được chọn
    selected_auc = {disease: auc for disease, auc in per_class_auc.items()}
    
    # Tính AUC trung bình
    auc_mean = sum(selected_auc.values()) / len(selected_auc) if selected_auc else 0.0
    
    # In AUC trung bình
    print(f"Domain {domain} - AUC trung bình: {auc_mean:.4f}")
    
    # In danh sách AUC của các bệnh
    print(f"\nDomain {domain} - Danh sách AUC của các bệnh:")
    for disease, auc in selected_auc.items():
        print(f"{disease}: {auc:.4f}")

def get_pretrained_model(cfg):
    model_path = "./results/best_model_jul5_11h30.pth"
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