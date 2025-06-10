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