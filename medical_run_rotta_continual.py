import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from torchvision import transforms

# Import các thành phần đã tạo
from core.optim import build_optimizer
from medical_continual_data_loader import ContinualDomainLoader # Đã có từ câu trả lời trước
from Medical.models import get_model, loadModelFor6Classes
from core.adapter.rotta_multilabel_adapter import RoTTAMultiLabel
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# FINETUNED_MODEL_PATH = "./Medical/results/finetuned_model_mobile_net_lr0001_latest.pth"
FINETUNED_MODEL_PATH = "./Medical/results/finetuned_model_resnet_jun25_22h40.pth"

# def get_pretrained_model(model_path, cfg):
#     print(f"Loading fine-tuned weights from: {model_path}")
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found at {model_path}. Please run the training script first.")
    
#     print(f"Found fine-tuned model at {model_path}")
#     # Load the pre-trained model architecture
#     model = get_model(cfg, useWeight=True)
#     # Load the fine-tuned weights
#     model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#     model.to(DEVICE)
#     print(f"Loaded fine-tuned model from {model_path}")
    
#     print("Fine-tuned model loaded successfully.")
#     return model

def get_pretrained_model(model_path, cfg):
    print(f"Loading fine-tuned weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run the training script first.")
    
    print(f"Found fine-tuned model at {model_path}")
    # Load the pre-trained model architecture
    # model = get_model(cfg, useWeight=True)
    # Load the fine-tuned weights
    # model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    # model.to(DEVICE)
    model = loadModelFor6Classes(cfg, model_path)
    print(f"Loaded fine-tuned model from {model_path}")
    
    print("Fine-tuned model loaded successfully.")
    return model

def main(cfg):
    device = torch.device(cfg.TRAINING.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Tải mô hình nguồn đã được fine-tune trên CheXpert
    model = get_pretrained_model(model_path=FINETUNED_MODEL_PATH, cfg=cfg)
    
    # 2. Khởi tạo RoTTA adapter
    optimizer = build_optimizer(cfg)
    tta_model = RoTTAMultiLabel(cfg, model).to(device)

    print('get_occupancy->get_occupancy', tta_model.mem.get_list_class_name())
    # 3. Tạo luồng dữ liệu liên tục
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Chuỗi các domain thay đổi theo từng batch
    # domain_sequence = ['vindr', 'nih14', 'padchest'] * 33 + ['vindr'] 
    domain_sequence = ['vindr', 'nih14'] * 50

    continual_loader = ContinualDomainLoader(
        cfg, 
        domain_sequence=domain_sequence, 
        batch_size=cfg.ADAPTER.BATCH_SIZE, 
        transform=eval_transform
    )
    
    # 4. Chạy quá trình thích ứng và đánh giá
    all_preds = {name: [] for name in continual_loader.datasets.keys()}
    all_labels = {name: [] for name in continual_loader.datasets.keys()}

    pbar = tqdm(continual_loader, total=len(continual_loader), desc="Continual Test-Time Adaptation")
    for data_package in pbar:
        if data_package['image'].numel() == 0: continue
        images, labels, domains = data_package['image'], data_package['label'], data_package['domain']
        images = images.to(device)
        
        # Quá trình TTA diễn ra bên trong lệnh forward này
        outputs = tta_model(images)
        
        current_domain = domains[0]
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds[current_domain].append(probs)
        all_labels[current_domain].append(labels.numpy())
        
        # pbar.set_postfix(domain=current_domain, bank_size=tta_model.mem.get_occupancy())
        pbar.set_postfix(domain=current_domain, bank_occupancy=tta_model.mem.get_occupancy())

    # 5. In kết quả cuối cùng
    print("\n--- Final Performance after Continual Adaptation ---")
    for domain_name in all_preds.keys():
        if not all_preds[domain_name]: continue
        
        preds = np.concatenate(all_preds[domain_name], axis=0)
        labels = np.concatenate(all_labels[domain_name], axis=0)
        
        auc_scores = []
        for i in range(labels.shape[1]):
            if len(np.unique(labels[:, i])) > 1:
                auc = roc_auc_score(labels[:, i], preds[:, i])
                auc_scores.append(auc)
        
        mean_auc = np.mean(auc_scores) if auc_scores else 0.0
        print(f"Mean AUC on {domain_name}: {mean_auc:.4f}")

if __name__ == "__main__":
    cfg = OmegaConf.load("Medical/configs/base_config.yaml")
    main(cfg)