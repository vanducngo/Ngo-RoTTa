import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from torchvision import transforms
import torch.optim as optim

# Import các thành phần đã tạo
from Medical.constants import COMMON_FINAL_LABEL_SET
from Medical.utils import get_pretrained_model, print_selected_auc_stats
from medical_continual_data_loader import ContinualDomainLoader # Đã có từ câu trả lời trước
from core.adapter.rotta_multilabel_adapter import RoTTAMultiLabelSelective

def main(cfg):
    device = torch.device(cfg.TRAINING.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Tải mô hình nguồn đã được fine-tune trên CheXpert
    model = get_pretrained_model(cfg=cfg)
    
    # 2. Khởi tạo RoTTA adapter
    optimizer_func = lambda params: optim.Adam(params, lr=cfg.ADAPTER.LR, weight_decay=cfg.OPTIM.WD)
    tta_model = RoTTAMultiLabelSelective(cfg, model, optimizer_func).to(device)

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
        
        all_probs = np.concatenate(all_preds[domain_name], axis=0)
        labels = np.concatenate(all_labels[domain_name], axis=0)
        
        auc_scores = {}
        valid_aucs = []
        for i, class_name in enumerate(COMMON_FINAL_LABEL_SET):
            if len(np.unique(labels[:, i])) > 1:  # Use concatenated labels
                try:
                    auc = roc_auc_score(labels[:, i], all_probs[:, i])
                    auc_scores[f"auc/{class_name}"] = auc  # Tên key phù hợp cho wandb
                    valid_aucs.append(auc)
                except ValueError:
                    pass
        
        print(f"auc_scores: {auc_scores}")
        print_selected_auc_stats(auc_scores, domain_name)

if __name__ == "__main__":
    cfg = OmegaConf.load("Medical/configs/base_config.yaml")
    main(cfg)