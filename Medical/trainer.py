import torch
import torch.nn as nn
from tqdm import tqdm
from utils import calculate_auc

def fine_tune(cfg, model, train_loader, device):
    """Fine-tune lại mô hình trên tập train của CheXpert."""
    print("\n>>> Starting fine-tuning on CheXpert...")
    
    model.to(device)
    model.train()
    
    # Dùng Binary Cross Entropy cho bài toán đa nhãn
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAINING.LEARNING_RATE)
    
    for epoch in range(cfg.TRAINING.EPOCHS):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.TRAINING.EPOCHS}")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # print ('Label:', labels)
            # if torch.isnan(images).any() or torch.isinf(images).any():
            #     print("NaN or Inf found in input images")
            # if torch.isnan(labels).any() or torch.isinf(labels).any():
            #     print("NaN or Inf found in labels")

            optimizer.zero_grad()
            
            outputs = model(images)

            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("NaN or Inf found in model outputs")          

            loss = criterion(outputs, labels)
            
            loss.backward()

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("NaN or Inf found in loss")

            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))
            
    print(">>> Fine-tuning finished.")
    return model

def evaluate(model, data_loader, device):
    """Đánh giá hiệu suất của mô hình trên một tập dữ liệu."""
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            
            outputs = model(images)
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    mean_auc = calculate_auc(all_preds, all_labels)
    return mean_auc