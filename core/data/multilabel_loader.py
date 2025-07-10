import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import pandas as pd
import os
from PIL import Image
import numpy as np

from ..utils.metrics import AUCProcessor

class SingleDomainDataset(Dataset):
    """
    Lớp Dataset cho một miền dữ liệu duy nhất.
    Giả định file CSV đã được tiền xử lý và có các cột nhãn.
    """
    def __init__(self, root_path, csv_name, domain_name, image_dir_name, labels_list, transform=None):
        self.root_dir = os.path.join(root_path, image_dir_name)
        self.transform = transform
        self.domain_name = domain_name
        self.labels_list = labels_list
        
        
        csv_path = os.path.join(root_path, csv_name)
        print(f'csv_path:{csv_path}')
        self.df = pd.read_csv(csv_path)
        
        # Xác định các thông số dựa trên tên domain
        self.image_col = 'image_id'
        self.path_prefix = ''
        if domain_name == 'vindr':
             self.path_prefix = '.png'
        if 'chexpert' in domain_name.lower():
            self.image_col = 'Path'
            self.root_dir = root_path
            
        print(f"Initialized '{domain_name}' dataset with {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row[self.image_col])
        
        # Xử lý đường dẫn cho Chexpert có thể là tuyệt đối hoặc tương đối
        if 'chexpert' in self.domain_name.lower() and not os.path.isabs(img_name):
             img_path = os.path.join(self.root_dir, img_name)
        elif 'chexpert' in self.domain_name.lower() and os.path.isabs(img_name):
             img_path = img_name
        else:
             img_path = os.path.join(self.root_dir, img_name + self.path_prefix)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: File not found at {img_path}. Returning None.")
            return None
            
        labels = torch.tensor(row[self.labels_list].values.astype('float'), dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        # Trả về domain index thay vì tên domain để dễ xử lý
        return {'image': image, 'label': labels, 'domain': self.domain_name}

def collate_fn_skip_none(batch):
    """Bỏ qua các mẫu bị lỗi (None) trong batch và collate."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        # Trả về tensor rỗng nếu cả batch đều lỗi
        return {'image': torch.empty(0), 'label': torch.empty(0), 'domain': []}
    return torch.utils.data.dataloader.default_collate(batch)


class ContinualDomainIterator:
    """
    Tạo ra một iterator để lặp qua dữ liệu từ nhiều domain một cách liên tục.
    Phiên bản này có thể được dùng với DataLoader của PyTorch.
    """
    def __init__(self, cfg):
        # Đọc các cấu hình từ file YAML
        self.domains_to_load = cfg.DATASET.TEST_DOMAINS
        self.labels_list = cfg.DATASET.LABELS_LIST
        self.transform = self._build_transforms(cfg)
        
        # Khởi tạo các dataset con cho mỗi domain
        self.datasets = {
            name: SingleDomainDataset(
                root_path=getattr(cfg.DATASET, f"{name.upper()}_PATH"),
                csv_name=getattr(cfg.DATASET, f"{name.upper()}_CSV"),
                domain_name=name,
                image_dir_name=getattr(cfg.DATASET, f"{name.upper()}_IMAGE_DIR", ""), # Thêm giá trị mặc định
                labels_list=self.labels_list,
                transform=self.transform
            )
            for name in self.domains_to_load
        }
        
        # Tạo một danh sách lớn chứa tất cả các mẫu
        self.all_samples = []
        for domain_name in self.domains_to_load:
            dataset = self.datasets[domain_name]
            for i in range(len(dataset)):
                self.all_samples.append((dataset, i))
                
        print(f"Continual iterator created with {len(self.all_samples)} total samples across {len(self.domains_to_load)} domains.")

    def _build_transforms(self, cfg):
        # Định nghĩa transform chuẩn cho ảnh y tế
        return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # transforms.GaussianBlur(kernel_size=BLUR_KERNEL_SIZE, sigma=BLUR_SIGMA),
        # transforms.Lambda(lambda x: torch.clamp(x + 0.005 * torch.randn_like(x), 0, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        dataset, sample_idx = self.all_samples[idx]
        return dataset[sample_idx]

# --- Hàm build_loader chính cho bài toán đa nhãn ---

def build_loader_multilabel(cfg):
    """
    Xây dựng DataLoader và Processor cho kịch bản Test-Time Adaptation đa nhãn, đa domain.
    """
    # 1. Khởi tạo một Dataset lớn chứa dữ liệu từ tất cả các domain test
    continual_dataset = ContinualDomainIterator(cfg)
    
    # 2. Tạo DataLoader từ dataset này
    # Sampler không cần thiết vì ContinualDomainIterator đã sắp xếp dữ liệu theo domain
    loader = DataLoader(
        continual_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,  # Dữ liệu đã được sắp xếp theo domain, không xáo trộn
        num_workers=cfg.LOADER.NUM_WORKS,
        collate_fn=collate_fn_skip_none, # Xử lý các ảnh bị lỗi
        pin_memory=True
    )
    
    # 3. Tạo processor để tính toán metric (AUC)
    # Giả sử số lượng lớp được định nghĩa trong config
    result_processor = AUCProcessor(num_classes=len(cfg.DATASET.LABELS_LIST))
    
    print("Multi-label, multi-domain loader and AUC processor are built successfully.")
    
    # Trả về loader và processor, giống cấu trúc hàm gốc
    return loader, result_processor