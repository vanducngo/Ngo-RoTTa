import os
import sys
# Lấy đường dẫn của thư mục gốc project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
from data_loader import get_chexpert_full_label_loaders
from train_chexpert import evaluate, fine_tune
from omegaconf import OmegaConf
import torch.nn as nn

from Base.models import get_model_chexpert

def main(cfg):
    device = torch.device(cfg.TRAINING.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_save_path = os.path.join(cfg.OUTPUT_DIR, "finetuned_model.pth")

    print("\n>>> Loading datasets...")
    train_loader, chexpert_test_loader = get_chexpert_full_label_loaders(cfg)

    if os.path.exists(model_save_path):
        print(f"Found fine-tuned model at {model_save_path}")
        model = get_model_chexpert(cfg)
        model.load_state_dict(torch.load(model_save_path))
        model.to(device)

        print(f"Loaded fine-tuned model from {model_save_path}")
    else:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        model = get_model_chexpert(cfg)
        model = fine_tune(cfg, model, train_loader, chexpert_test_loader, device)

        # save fine-tune model
        model_save_path = os.path.join(cfg.OUTPUT_DIR, "finetuned_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Fine-tuned model saved to {model_save_path}")

    print("\n--- Performance Evaluation ---")
    print("Evaluating on CheXpert test set (In-Domain)...")
    criterion = nn.BCEWithLogitsLoss(pos_weight=None)
    mean_valid_auc, epoch_valid_loss, per_class_auc = evaluate(model, chexpert_test_loader, device, criterion)
    print(f"Valid Loss: {epoch_valid_loss:.4f} | Valid AUC: {mean_valid_auc:.4f}")
    print(f"Per class auc: \n {per_class_auc}")
        
if __name__ == "__main__":
    cfg = OmegaConf.load('configs/base_config.yaml')
    main(cfg)