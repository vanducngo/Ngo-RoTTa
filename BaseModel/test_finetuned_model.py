import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from omegaconf import OmegaConf

from constants import COMMON_FINAL_LABEL_SET, TRAINING_LABEL_SET
from data_loader import get_chexpert_full_label_loaders, get_data_loaders_nih14, get_data_loaders_padchest, get_data_loaders_vindr
from utils import DEVICE, evaluate_model, get_pretrained_model

def print_selected_auc_stats(per_class_auc, domain = 'X'):
    # Lấy AUC của các bệnh được chọn
    selected_auc = {f'auc/{disease}': per_class_auc[f'auc/{disease}'] for disease in COMMON_FINAL_LABEL_SET}
    
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

def main():
    print(f"Using device: {DEVICE}")

    cfg = OmegaConf.load('configs/base_config.yaml')
    
    # 1. Tải mô hình
    model = get_pretrained_model(cfg=cfg)
    
    # 2. Chạy đánh giá
    
    # Base 
    _, chexpert_test_loader =  get_chexpert_full_label_loaders(cfg)
    # mean_valid_auc, epoch_valid_loss, per_class_auc = evaluate(model, chexpert_test_loader, DEVICE)
    # print_selected_auc_stats(per_class_auc, "CheXpert")

    mean_valid_auc, epoch_valid_loss, per_class_auc = evaluate_model(model, chexpert_test_loader)
    print_selected_auc_stats(per_class_auc, "CheXpert")
    
    # VinDr CXR 
    vindr_test_loader =  get_data_loaders_vindr(cfg)
    mean_valid_auc, epoch_valid_loss, per_class_auc = evaluate_model(model, vindr_test_loader)
    print_selected_auc_stats(per_class_auc, 'VinDr')
    
    # NIH 14 dataset
    nih14_test_loader = get_data_loaders_nih14(cfg)
    mean_valid_auc, epoch_valid_loss, per_class_auc = evaluate_model(model, nih14_test_loader)
    print_selected_auc_stats(per_class_auc, "nih-14")
    
    # PadChest dataset
    # padchest_test_loader = get_data_loaders_padchest(cfg)
    # mean_valid_auc, epoch_valid_loss, per_class_auc = evaluate_model(model, padchest_test_loader)
    # print_selected_auc_stats(per_class_auc, "padchest")

if __name__ == "__main__":
    main()