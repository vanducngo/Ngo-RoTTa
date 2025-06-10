import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CheXpertDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.root_dir = cfg.DATA.CHEXPERT_PATH
        
        if mode == 'train':
            csv_file = os.path.join(self.root_dir, cfg.DATA.CHEXPERT_TRAIN_CSV)
        else: # test mode
            csv_file = os.path.join(self.root_dir, cfg.DATA.CHEXPERT_TEST_CSV)
            
        self.df = pd.read_csv(csv_file)
        
        # Chỉ giữ lại các cột là các bệnh chung
        self.df = self.df[['Path'] + cfg.COMMON_DISEASES]
        
        # Xử lý các giá trị NaN và -1 (uncertain)
        # Cách đơn giản: coi NaN và -1 là 0 (âm tính)
        self.df = self.df.fillna(0)
        self.df = self.df.replace(-1.0, 0)
        
        print(f"Loaded CheXpert {mode} dataset with {len(self.df)} samples.")
        
        # Định nghĩa các phép biến đổi ảnh
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        # CheXpert là ảnh xám, cần chuyển sang 3 kênh cho ResNet
        image = Image.open(img_path).convert('RGB')
        
        labels = self.df.iloc[idx, 1:].values.astype('float')
        labels = torch.tensor(labels, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels

class VinDrCXRDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.root_dir = os.path.join(cfg.DATA.VINDR_PATH, "test")
        csv_file = os.path.join(cfg.DATA.VINDR_PATH, cfg.DATA.VINDR_TEST_CSV)
        
        self.df = pd.read_csv(csv_file)
        
        # Chỉ giữ lại các bệnh chung
        # Lưu ý: Tên cột trong VinDr-CXR có thể khác, bạn cần kiểm tra và ánh xạ lại
        # Giả sử tên cột đã được chuẩn hóa giống COMMON_DISEASES
        self.df = self.df[['image_id'] + cfg.COMMON_DISEASES]
        
        # VinDr-CXR thường đã được gán nhãn 0/1, không cần xử lý -1
        print(f"Loaded VinDr-CXR test dataset with {len(self.df)} samples.")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Đường dẫn ảnh trong VinDr-CXR có thể cần xử lý khác
        img_path = os.path.join(self.root_dir, self.df.iloc[idx, 0] + ".jpg")
        image = Image.open(img_path).convert('RGB')
        
        labels = self.df.iloc[idx, 1:].values.astype('float')
        labels = torch.tensor(labels, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels

def get_data_loaders(cfg):
    """Tạo các DataLoader cho việc huấn luyện và kiểm tra."""
    
    # train_dataset = CheXpertDataset(cfg, mode='train')
    # train_subset = torch.utils.data.Subset(train_dataset, range(len(train_dataset) // 10)) # Lấy một subset nhỏ để train nhanh hơn cho mục đích demo
    # train_loader = DataLoader(train_subset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=True)
    
    # chexpert_test_dataset = CheXpertDataset(cfg, mode='test')
    # chexpert_test_loader = DataLoader(chexpert_test_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=False)
    
    train_loader=None
    chexpert_test_loader = None

    
    vindr_test_dataset = VinDrCXRDataset(cfg)
    vindr_test_loader = DataLoader(vindr_test_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=False)
    # vindr_test_loader = None #TODO:
    
    return train_loader, chexpert_test_loader, vindr_test_loader