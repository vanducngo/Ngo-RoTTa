import torch
from sklearn.metrics import roc_auc_score
import numpy as np

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