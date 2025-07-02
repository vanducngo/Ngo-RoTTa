import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from torchvision import transforms
import os

from Medical.models import get_model
from medical_continual_data_loader import ContinualDomainLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# FINETUNED_MODEL_PATH = "./Medical/results/finetuned_model_mobile_net_lr0001_latest.pth"
FINETUNED_MODEL_PATH = "./Medical/results/finetuned_model_resnet_jun25_22h40.pth"

def get_pretrained_model(model_path, cfg):
    print(f"Loading fine-tuned weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run the training script first.")
    
    print(f"Found fine-tuned model at {model_path}")
    # Load the pre-trained model architecture
    model = get_model(cfg, useWeight=True)
    # Load the fine-tuned weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    print(f"Loaded fine-tuned model from {model_path}")
    
    print("Fine-tuned model loaded successfully.")
    return model

def evaluate_continual_zero_shot(cfg, model, continual_loader, device):
    """
    Đánh giá mô hình trên một luồng dữ liệu liên tục mà không có DA.
    Hàm này sẽ lặp qua loader, thu thập dự đoán và nhãn cho từng domain.
    """
    model.to(device)
    model.eval()

    all_preds = {name: [] for name in continual_loader.datasets.keys()}
    all_labels = {name: [] for name in continual_loader.datasets.keys()}
    
    print("\n>>> Starting Zero-Shot Continual Domain Evaluation (No Adaptation)...")
    pbar = tqdm(continual_loader, total=len(continual_loader), desc="Evaluating across domains")
    
    with torch.no_grad(): # Không cần tính gradient
        for data_package in pbar:
            if data_package['image'].numel() == 0:
                continue
            
            images, labels, domains = data_package['image'], data_package['label'], data_package['domain']
            images = images.to(device)
            
            # Chỉ thực hiện forward pass, không có bước adapt
            outputs = model(images)
            
            # Thu thập kết quả
            current_domain = domains[0]
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds[current_domain].append(probs)
            all_labels[current_domain].append(labels.numpy())
            
            pbar.set_postfix(domain=current_domain)
            
    # Tính toán và in kết quả cuối cùng
    print("\n--- Zero-Shot Performance (AUC) per Domain ---")
    final_results = {}
    for domain_name in all_preds.keys():
        if not all_preds[domain_name]:
            print(f"No samples evaluated for domain: {domain_name}")
            continue
        
        preds = np.concatenate(all_preds[domain_name], axis=0)
        labels = np.concatenate(all_labels[domain_name], axis=0)
        
        auc_scores = []
        for i in range(labels.shape[1]):
            # Chỉ tính AUC nếu có cả nhãn dương và âm
            if len(np.unique(labels[:, i])) > 1:
                try:
                    auc = roc_auc_score(labels[:, i], preds[:, i])
                    auc_scores.append(auc)
                except ValueError:
                    pass
        
        mean_auc = np.mean(auc_scores) if auc_scores else 0.0
        final_results[domain_name] = mean_auc
        print(f"Mean AUC on {domain_name}: {mean_auc:.4f}")
        
    return final_results


def main(cfg):
    device = torch.device(cfg.TRAINING.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n>>> Loading the fine-tuned source model...")
    model = get_pretrained_model(model_path=FINETUNED_MODEL_PATH, cfg=cfg)

    # 2. Tạo luồng dữ liệu liên tục từ các miền đích
    print("\n>>> Preparing the continual domain data stream...")
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # domain_sequence = (['vindr', 'nih14', 'padchest'] * 34)[:100] # Tạo chuỗi 100 batch luân phiên
    domain_sequence = ['vindr', 'nih14'] * 50
    
    continual_loader = ContinualDomainLoader(
        cfg, 
        domain_sequence=domain_sequence, 
        batch_size=cfg.ADAPTER.BATCH_SIZE, 
        transform=eval_transform
    )
    
    # 3. Chạy đánh giá không có Domain Adaptation
    evaluate_continual_zero_shot(cfg, model, continual_loader, device)

if __name__ == "__main__":
    try:
        cfg = OmegaConf.load("Medical/configs/base_config.yaml")
        main(cfg)
    except FileNotFoundError:
        print("Error: `configs yaml` not found.")
        print("Please create a configuration file for the continual adaptation experiment.")