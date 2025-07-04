import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image

from constants import COMMON_FINAL_LABEL_SET
from data_mapping import map_vindr_labels, map_chestxray14_labels, map_padchest_labels

class SingleDomainDataset(Dataset):
    def __init__(self, root_path, csv_path, domain_name, label_mapper, transform=None):
        self.root_dir = os.path.join(root_path, 'images')
        self.transform = transform
        self.domain_name = domain_name
    
        self.df = pd.read_csv(os.path.join(root_path, csv_path))
        
        self.image_col = 'image_id'
        self.path_prefix = ''
        if domain_name == 'vindr':
             self.path_prefix = '.png'
        
        print(f"Initialized '{domain_name}' dataset with {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row[self.image_col] + self.path_prefix)
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            return None
            
        labels = torch.tensor(row[COMMON_FINAL_LABEL_SET].values.astype('float'), dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return {'image': image, 'label': labels, 'domain': self.domain_name}

class ContinualDomainLoader:
    """
    Tạo ra một iterator để lặp qua các batch từ các domain khác nhau theo chuỗi.
    """
    def __init__(self, cfg, domain_sequence, batch_size, transform):
        self.datasets = {}
        # Khởi tạo các dataset con cho mỗi domain
        for domain_name in set(domain_sequence): # Chỉ tạo dataset 1 lần cho mỗi domain
            if domain_name == 'vindr':
                root = cfg.DATA.VINDR_PATH
                csv = cfg.DATA.VINDR_CSV
                mapper = map_vindr_labels
            elif domain_name == 'nih14':
                root = cfg.DATA.CHESTXRAY14_PATH
                csv = cfg.DATA.CHESTXRAY14_CSV
                mapper = map_chestxray14_labels
            elif domain_name == 'padchest':
                root = cfg.DATA.PADCHEST_PATH
                csv = cfg.DATA.PADCHEST_CSV
                mapper = map_padchest_labels
            else:
                raise ValueError(f"Unknown domain: {domain_name}")
                
            self.datasets[domain_name] = SingleDomainDataset(root, csv, domain_name, mapper, transform)
        
        self.domain_sequence = domain_sequence
        self.batch_size = batch_size
        self.num_domains = len(self.domain_sequence)

    def __iter__(self):
        self.domain_index = 0
        self.dataset_indices = {name: 0 for name in self.datasets}
        return self

    def __next__(self):
        if self.domain_index >= self.num_domains:
            raise StopIteration

        # Lấy domain hiện tại
        current_domain_name = self.domain_sequence[self.domain_index]
        current_dataset = self.datasets[current_domain_name]
        
        # Lấy batch từ domain hiện tại
        start_idx = self.dataset_indices[current_domain_name]
        end_idx = start_idx + self.batch_size
        
        # Nếu hết dữ liệu trong domain này, quay lại từ đầu (hoặc dừng)
        if start_idx >= len(current_dataset):
            # Đã duyệt hết chuỗi, dừng lại
            raise StopIteration
            # Hoặc nếu muốn lặp vô hạn trong 1 epoch:
            # self.dataset_indices[current_domain_name] = 0
            # start_idx = 0
            # end_idx = start_idx + self.batch_size

        batch_data = []
        for i in range(start_idx, min(end_idx, len(current_dataset))):
            sample = current_dataset[i]
            if sample is not None:
                batch_data.append(sample)
        
        # Cập nhật con trỏ
        self.dataset_indices[current_domain_name] = end_idx
        self.domain_index += 1
        
        if not batch_data:
             return self.__next__() # Bỏ qua batch rỗng và lấy batch tiếp theo

        # Gộp batch
        images = torch.stack([s['image'] for s in batch_data])
        labels = torch.stack([s['label'] for s in batch_data])
        domains = [s['domain'] for s in batch_data]

        return {'image': images, 'label': labels, 'domain': domains}

    def __len__(self):
        return self.num_domains # Số lượng batch sẽ bằng số lượng domain trong chuỗi