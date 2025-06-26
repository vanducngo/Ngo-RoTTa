import torch
from data_loader import get_data_loaders_cheXpert
from trainer import evaluate, fine_tune
from omegaconf import OmegaConf
import os
import torch.nn as nn

from models import get_model


def main(cfg):
    # Thiết lập
    device = torch.device(cfg.TRAINING.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_save_path = os.path.join(cfg.OUTPUT_DIR, "finetuned_model.pth")

    # Tải dữ liệu
    print("\n>>> Loading datasets...")
    train_loader, chexpert_test_loader = get_data_loaders_cheXpert(cfg)

    if os.path.exists(model_save_path):
        print(f"Found fine-tuned model at {model_save_path}")
        # Step 1: Load the pre-trained model architecture
        model = get_model(cfg)
        # Load the fine-tuned weights
        model.load_state_dict(torch.load(model_save_path))
        model.to(device)
        print(f"Loaded fine-tuned model from {model_save_path}")
    else:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        # Bước 1: Tải Pre-trained model
        model = get_model(cfg)
        
        # Bước 2: Fine-tune mô hình trên CheXpert
        model = fine_tune(cfg, model, train_loader, chexpert_test_loader, device)
        
        # Lưu lại model đã fine-tune
        model_save_path = os.path.join(cfg.OUTPUT_DIR, "finetuned_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Fine-tuned model saved to {model_save_path}")

        # Bước 3: Kiểm tra hiệu suất
        print("\n--- Performance Evaluation ---")
        
        # Đánh giá trên tập test của CheXpert (In-Domain Performance)
        print("Evaluating on CheXpert test set (In-Domain)...")
        criterion = nn.BCEWithLogitsLoss()
        mean_valid_auc, epoch_valid_loss, per_class_auc = evaluate(model, chexpert_test_loader, device, criterion)
        print(f"Valid Loss: {epoch_valid_loss:.4f} | Valid AUC: {mean_valid_auc:.4f}")
        print(f"Per class auc: \n {per_class_auc}")
    
    # Đánh giá trên tập test của VinDr-CXR (Out-of-Domain Performance)
    # print("\nEvaluating on VinDr-CXR test set (Out-of-Domain)...")
    # vindr_auc = evaluate(model, vindr_test_loader, device)
    # print(f"VinDr-CXR Test Mean AUC: {vindr_auc:.4f}")

    '''
    # Bước 5: So sánh và đánh giá Domain Shift
    print("\n--- Domain Shift Analysis ---")
    performance_drop = chexpert_auc - vindr_auc
    print(f"In-Domain AUC (CheXpert): {chexpert_auc:.4f}")
    print(f"Out-of-Domain AUC (VinDr-CXR): {vindr_auc:.4f}")
    print(f"Performance Drop due to Domain Shift: {performance_drop:.4f} ({performance_drop/chexpert_auc:.2%})")
    
    if performance_drop > 0.05:
        print("\nConclusion: Có sự sụt giảm hiệu suất đáng kể, cho thấy sự tồn tại của Domain Distribution Shift giữa CheXpert và VinDr-CXR.")
    else:
        print("\nConclusion: Hiệu suất khá tương đồng, mô hình tổng quát hóa tốt giữa hai domain.")

    '''
if __name__ == "__main__":
    # Tải cấu hình
    cfg = OmegaConf.load('configs/base_config.yaml')
    main(cfg)