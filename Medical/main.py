import torch
from data_loader import get_data_loaders
from trainer import evaluate, fine_tune
from omegaconf import OmegaConf
import os
import numpy as np
from models import get_model
from data_mapping import FINAL_LABEL_SET

def compute_class_weights(full_train_dataset, device):
    """
    Tính trọng số cho loss dựa trên tỷ lệ nghịch của tần suất lớp dương tính.
    """
    print(">>> Computing class weights for weighted loss...")
    df = full_train_dataset.df
    # Tính số ca dương tính cho mỗi bệnh
    pos_counts = df[FINAL_LABEL_SET].sum(axis=0).values
    # Tổng số mẫu
    total_samples = len(df)
    
    # Tính trọng số: weight = số ca âm / số ca dương
    weights = (total_samples - pos_counts) / (pos_counts + 1e-6)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    print(f"Positive counts: {pos_counts.tolist()}")
    print(f"Computed class weights: {weights.tolist()}")
    return weights

def main(cfg):
    device = torch.device(cfg.TRAINING.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_save_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.ARCH}_finetuned.pth")

    print("\n>>> Loading datasets...")
    train_loader, valid_loader, chexpert_test_loader, vindr_test_loader = get_data_loaders(cfg)

    # Tính trọng số cho loss nếu được cấu hình
    class_weights = None
    if cfg.TRAINING.USE_WEIGHTED_LOSS:
        # Cần dataset gốc để tính trọng số chính xác
        full_train_dataset = train_loader.dataset.dataset 
        class_weights = compute_class_weights(full_train_dataset, device)
        
    criterion_eval = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # Kiểm tra xem mô hình đã được huấn luyện chưa
    if os.path.exists(model_save_path):
        print(f"\nFound pre-trained fine-tuned model at {model_save_path}")
        model = get_model(cfg)
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print("Loaded fine-tuned model successfully.")
    else:
        print(f"\nNo fine-tuned model found. Starting a new training session...")
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        model = get_model(cfg)
        model = fine_tune(cfg, model, train_loader, valid_loader, device, class_weights=class_weights)
        
        torch.save(model.state_dict(), model_save_path)
        print(f"Fine-tuned model saved to {model_save_path}")

    model.to(device)

    print("\n--- Final Performance Evaluation ---")
    
    # Đánh giá trên tập test của CheXpert (In-Domain)
    print("\nEvaluating on CheXpert test set (In-Domain)...")
    chexpert_auc, _ = evaluate(model, chexpert_test_loader, device, criterion_eval)
    print(f"==> CheXpert Test Mean AUC: {chexpert_auc:.4f}")
    
    # Đánh giá trên tập test của VinDr-CXR (Out-of-Domain)
    print("\nEvaluating on VinDr-CXR test set (Out-of-Domain)...")
    vindr_auc, _ = evaluate(model, vindr_test_loader, device, criterion_eval)
    print(f"==> VinDr-CXR Test Mean AUC: {vindr_auc:.4f}")

if __name__ == "__main__":
    cfg = OmegaConf.load('configs/config.yaml')
    main(cfg)