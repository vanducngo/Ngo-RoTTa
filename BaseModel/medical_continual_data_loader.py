import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
import numpy as np

# Giả định các file này có thể được import
from constants import COMMON_FINAL_LABEL_SET
# Vì bỏ label_mapper, chúng ta giả định file CSV đã được tiền xử lý
# để có các cột khớp với COMMON_FINAL_LABEL_SET

class SingleDomainDataset(Dataset):
    """
    Lớp Dataset cho một miền dữ liệu duy nhất.
    Giả định file CSV đã được tiền xử lý.
    """
    def __init__(self, root_path, csv_name, domain_name, image_dir_name, transform=None):
        self.root_dir = os.path.join(root_path, image_dir_name)
        self.transform = transform
        self.domain_name = domain_name
        
        csv_path = os.path.join(root_path, csv_name)
        self.df = pd.read_csv(csv_path)
        
        # Xác định các thông số dựa trên tên domain
        self.image_col = 'image_id'
        
        self.path_prefix = ''
        if domain_name == 'vindr':
             self.path_prefix = '.png'
             
        if 'chexpert' in domain_name.lower():
            self.image_col = 'Path'
            self.path_prefix = ''
            self.root_dir = root_path # Chexpert có đường dẫn đầy đủ
            
        print(f"Initialized '{domain_name}' dataset with {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row[self.image_col]
        
        img_path = os.path.join(self.root_dir, img_name + self.path_prefix)
        # Xử lý đường dẫn tuyệt đối cho Chexpert
        if 'chexpert' in self.domain_name.lower() and not os.path.isabs(img_name):
             img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: File not found at {img_path}. Returning None.")
            return None
            
        labels = torch.tensor(row[COMMON_FINAL_LABEL_SET].values.astype('float'), dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return {'image': image, 'label': labels, 'domain': self.domain_name}

def collate_fn_skip_none(batch):
    """Bỏ qua các mẫu bị lỗi (None) trong batch."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return {'image': torch.empty(0), 'label': torch.empty(0), 'domain': []}
    return torch.utils.data.dataloader.default_collate(batch)


class ContinualDomainLoader:
    """
    Tạo ra một iterator để lặp qua TẤT CẢ dữ liệu từ nhiều domain,
    luân phiên theo từng batch cho đến khi mọi domain được duyệt hết.
    """
    def __init__(self, cfg, domains_to_load, batch_size, transform):
        self.batch_size = batch_size
        self.domain_names = domains_to_load # Ví dụ: ['vindr', 'nih14', 'padchest']
        
        # Khởi tạo các dataset con cho mỗi domain
        self.datasets = {
            name: SingleDomainDataset(
                root_path=getattr(cfg.DATA, f"{name.upper()}_PATH"),
                csv_name=getattr(cfg.DATA, f"{name.upper()}_CSV"),
                domain_name=name,
                image_dir_name=getattr(cfg.DATA, f"{name.upper()}_IMAGE_DIR"),
                transform=transform
            )
            for name in self.domain_names
        }

        # Tính toán tổng số batch sẽ được tạo ra
        self.total_batches = sum(
            np.ceil(len(ds) / self.batch_size) for ds in self.datasets.values()
        )
        print(f"Continual Loader will generate approximately {int(self.total_batches)} batches in total.")

    def __iter__(self):
        # Con trỏ cho vị trí hiện tại trong mỗi dataset
        self.dataset_indices = {name: 0 for name in self.domain_names}
        # Cờ để theo dõi domain nào đã hết dữ liệu
        self.finished_domains = {name: False for name in self.domain_names}
        # Con trỏ cho domain hiện tại sẽ được lấy batch
        self.current_domain_idx_ptr = 0
        return self

    def __next__(self):
        # Vòng lặp để tìm domain tiếp theo còn dữ liệu
        for _ in range(len(self.domain_names)):
            # Lấy tên domain hiện tại và luân phiên
            domain_name = self.domain_names[self.current_domain_idx_ptr]
            
            # Chuyển con trỏ domain cho lần gọi __next__ tiếp theo
            self.current_domain_idx_ptr = (self.current_domain_idx_ptr + 1) % len(self.domain_names)
            
            # Nếu domain này đã hết dữ liệu, bỏ qua
            if self.finished_domains[domain_name]:
                continue

            # Nếu domain này còn dữ liệu, lấy một batch
            current_dataset = self.datasets[domain_name]
            start_idx = self.dataset_indices[domain_name]
            end_idx = start_idx + self.batch_size
            
            batch_data = []
            for i in range(start_idx, min(end_idx, len(current_dataset))):
                sample = current_dataset[i]
                if sample is not None:
                    batch_data.append(sample)
            
            # Cập nhật con trỏ cho dataset này
            self.dataset_indices[domain_name] = end_idx

            # Kiểm tra xem dataset này đã hết chưa
            if self.dataset_indices[domain_name] >= len(current_dataset):
                self.finished_domains[domain_name] = True
            
            # Nếu batch không rỗng, trả về
            if batch_data:
                images = torch.stack([s['image'] for s in batch_data])
                labels = torch.stack([s['label'] for s in batch_data])
                domains = [s['domain'] for s in batch_data]
                return {'image': images, 'label': labels, 'domain': domains}

        # Nếu vòng lặp kết thúc mà không tìm thấy domain nào còn dữ liệu
        # -> tất cả đã xong -> dừng lặp
        raise StopIteration

    def __len__(self):
        return int(self.total_batches)