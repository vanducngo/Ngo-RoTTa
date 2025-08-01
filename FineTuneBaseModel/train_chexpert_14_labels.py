import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import copy
from constants import TRAINING_LABEL_SET
import wandb
from omegaconf import OmegaConf
import os

# Giả định data_mapping.py đã được import

def evaluate(model, data_loader, device, criterion):
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

def fine_tune(cfg, model, train_loader, valid_loader, device, class_weights=None):
    """
    Fine-tune mô hình với validation, early stopping, và logging bằng wandb.
    """
    print("\n>>> Starting fine-tuning on CheXpert...")

    # Khởi tạo wandb run
    wandb.init(
        project="chexpert-finetuning",
        config=OmegaConf.to_container(cfg, resolve=True), # Log toàn bộ config
        name=f"{cfg.MODEL.ARCH}-lr{cfg.TRAINING.LEARNING_RATE}-bs{cfg.TRAINING.BATCH_SIZE}"
    )
    wandb.watch(model, log="all", log_freq=100)
    
    model.to(device)
    
    # Sử dụng BCEWithLogitsLoss để ổn định hơn
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAINING.LEARNING_RATE, weight_decay=cfg.TRAINING.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    
    best_auc = 0.0
    best_model_state = None
    early_stop_counter = 0
    patience = cfg.TRAINING.EARLY_STOPPING_PATIENCE
    
    for epoch in range(cfg.TRAINING.EPOCHS):
        model.train() # Đặt model ở chế độ train
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.TRAINING.EPOCHS} [Training]")
        
        for step, (images, labels) in enumerate(pbar):
            if images.numel() == 0: continue
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Log loss của từng step lên wandb
            if step % 50 == 0: # Log mỗi 50 steps
                wandb.log({"train/step_loss": loss.item()})

        # Tính loss trung bình của epoch
        epoch_train_loss = running_loss / len(train_loader)
        
        # Đánh giá trên tập xác thực
        mean_valid_auc, epoch_valid_loss, per_class_auc = evaluate(model, valid_loader, device, criterion)
        
        print(f"Epoch {epoch+1} | Train Loss: {epoch_train_loss:.4f} | Valid Loss: {epoch_valid_loss:.4f} | Valid AUC: {mean_valid_auc:.4f}")
        print(f"Per class auc: \n {per_class_auc}")

        # Log các chỉ số của epoch lên wandb
        log_metrics = {
            "epoch": epoch + 1,
            "train/epoch_loss": epoch_train_loss,
            "valid/epoch_loss": epoch_valid_loss,
            "valid/mean_auc": mean_valid_auc,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        log_metrics.update(per_class_auc) # Thêm AUC của từng lớp vào dict
        wandb.log(log_metrics)
        
        # Cập nhật scheduler
        scheduler.step(mean_valid_auc)
        
        # Logic Early Stopping
        if mean_valid_auc > best_auc:
            print(f"Validation AUC improved from {best_auc:.4f} to {mean_valid_auc:.4f}. Saving model...")
            best_auc = mean_valid_auc
            best_model_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
            # Lưu checkpoint tốt nhất lên wandb
            torch.save(best_model_state, os.path.join(wandb.run.dir, "best_model.pth"))
        else:
            early_stop_counter += 1
            print(f"No improvement in validation AUC for {early_stop_counter} epochs.")
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs of no improvement.")
                break
        
        torch.cuda.empty_cache()
    
    # Kết thúc wandb run
    wandb.finish()
    
    # Tải lại trọng số của mô hình tốt nhất
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\n>>> Fine-tuning finished. Best validation AUC: {best_auc:.4f}")
    return model