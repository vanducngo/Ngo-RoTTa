import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import copy

# Giả định data_mapping.py đã được import
from data_mapping import FINAL_LABEL_SET

def fine_tune(cfg, model, train_loader, valid_loader, device, class_weights=None):
    """
    Fine-tune mô hình trên CheXpert với early stopping, scheduler, và weighted loss.
    """
    print("\n>>> Starting fine-tuning on CheXpert...")
    
    model.to(device)
    
    # Sử dụng BCEWithLogitsLoss để tăng tính ổn định số học
    # pos_weight giúp xử lý mất cân bằng lớp
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAINING.LEARNING_RATE, weight_decay=cfg.TRAINING.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
    
    best_auc = 0.0
    best_model_state = None
    early_stop_counter = 0
    patience = cfg.TRAINING.EARLY_STOPPING_PATIENCE
    
    for epoch in range(cfg.TRAINING.EPOCHS):
        model.train() # Đặt model ở chế độ train
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.TRAINING.EPOCHS} [Training]")
        
        for images, labels in pbar:
            if images.numel() == 0: continue # Bỏ qua batch rỗng
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Đánh giá trên tập xác thực
        valid_auc, valid_loss = evaluate(model, valid_loader, device, criterion)
        print(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f} | Valid Loss: {valid_loss:.4f} | Valid AUC: {valid_auc:.4f}")
        
        # Cập nhật scheduler
        scheduler.step(valid_auc)
        
        # Logic Early Stopping
        if valid_auc > best_auc:
            print(f"Validation AUC improved from {best_auc:.4f} to {valid_auc:.4f}. Saving model...")
            best_auc = valid_auc
            # Dùng deepcopy để đảm bảo trạng thái không bị thay đổi
            best_model_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement in validation AUC for {early_stop_counter} epochs.")
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs of no improvement.")
                break
    
    # Tải lại trọng số của mô hình tốt nhất
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\n>>> Fine-tuning finished. Best validation AUC: {best_auc:.4f}")
    return model

def evaluate(model, data_loader, device, criterion):
    """
    Đánh giá mô hình và trả về mean AUC và loss trung bình.
    """
    model.eval() # Đặt model ở chế độ eval
    
    all_probs = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            if images.numel() == 0: continue
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Tính AUC
    auc_scores = []
    for i in range(all_labels.shape[1]):
        # Chỉ tính AUC nếu có cả nhãn dương và âm
        if len(np.unique(all_labels[:, i])) > 1:
            try:
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                auc_scores.append(auc)
            except ValueError:
                pass # Bỏ qua nếu có lỗi
                
    mean_auc = np.mean(auc_scores) if auc_scores else 0.0
    avg_loss = total_loss / len(data_loader.dataset)
    
    return mean_auc, avg_loss