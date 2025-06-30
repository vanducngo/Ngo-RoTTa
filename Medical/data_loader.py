import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

from data_mapping import COMMON_DISEASES, map_chexpert_labels

class MultiSourceDataset(Dataset):
    # dataset_name: chexpert, vindr, padchest, nih14
    def __init__(self, cfg, dataset_name, mode='train', transform=None):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.mode = mode # Lưu lại mode để sử dụng
        self.transform = transform
        
        # Tải và xử lý DataFrame dựa trên tên bộ dữ liệu
        if self.dataset_name == 'chexpert':
            csv_path = os.path.join(cfg.DATA.CHEXPERT_PATH, 
                                    cfg.DATA.CHEXPERT_TRAIN_CSV if mode == 'train' else cfg.DATA.CHEXPERT_TEST_CSV)
            raw_df = pd.read_csv(csv_path)
            self.df = map_chexpert_labels(raw_df)
            self.root_dir = cfg.DATA.CHEXPERT_PATH
            self.image_col = 'Path'
        elif self.dataset_name == 'vindr':
            csv_path = os.path.join(cfg.DATA.VINDR_PATH, cfg.DATA.VINDR_EVAL_CSV)
            self.df = pd.read_csv(csv_path)
            self.root_dir = os.path.join(cfg.DATA.VINDR_PATH, cfg.DATA.VINDR_IMAGE_DIR)
            self.image_col = 'image_id'
        elif self.dataset_name == 'nih14':
            csv_path = os.path.join(cfg.DATA.CHESTXRAY14_PATH, cfg.DATA.CHESTXRAY14_CSV)
            self.df = pd.read_csv(csv_path)
            self.root_dir = os.path.join(cfg.DATA.CHESTXRAY14_PATH, 'images') 
            self.image_col = 'image_id'
        elif self.dataset_name == 'padchest':
            csv_path = os.path.join(cfg.DATA.PADCHEST_PATH, cfg.DATA.PADCHEST_CSV)
            self.df = pd.read_csv(csv_path)
            self.root_dir = os.path.join(cfg.DATA.PADCHEST_PATH, 'images') 
            self.image_col = 'image_id'
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
        print(f"Loaded and mapped {self.dataset_name} ({mode}) with {len(self.df)} samples.")
        print(f"Common diseases being used: {COMMON_DISEASES}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Lấy tên file/đường dẫn ảnh từ dataframe
        img_name = self.df.iloc[idx][self.image_col]
        img_path = os.path.join(self.root_dir, img_name)
            
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: File not found at {img_path}. Skipping.")
            return torch.empty(0), torch.empty(0)
        except Exception as e:
            print(f"Error reading {img_path}: {e}. Skipping.")
            return torch.empty(0), torch.empty(0)

        # Lấy nhãn từ các cột bệnh chung
        labels = self.df.iloc[idx][COMMON_DISEASES].values.astype('float')
        labels = torch.tensor(labels, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels

# Hàm helper để bỏ qua các mẫu bị lỗi
def collate_fn(batch):
    batch = list(filter(lambda x: x[0].numel() > 0, batch))
    if not batch:
        return torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)

def get_data_loaders_cheXpert(cfg):   
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transformTrain = transforms.Compose([
        transforms.Resize((224, 224)),
        
        # Thêm các phép biến đổi hình học mạnh hơn
        transforms.RandomRotation(15),  # Tăng góc xoay từ 10 lên 15 độ
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Dịch chuyển ảnh ngẫu nhiên 10%
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Cắt ngẫu nhiên vẫn giữ nguyên
        transforms.RandomHorizontalFlip(p=0.5), # Lật ngang là một phép rất hiệu quả
        
        # Thêm các phép biến đổi màu sắc mạnh hơn
        transforms.ColorJitter(brightness=0.3, contrast=0.3), # Tăng độ sáng/tương phản từ 0.1 lên 0.3
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MultiSourceDataset(cfg, dataset_name='chexpert', mode='train', transform=transformTrain)
    train_subset = torch.utils.data.Subset(train_dataset, range(len(train_dataset)))
    train_loader = DataLoader(train_subset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    chexpert_test_dataset = MultiSourceDataset(cfg, dataset_name='chexpert', mode='test', transform=transform)
    chexpert_test_loader = DataLoader(chexpert_test_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)

    return train_loader, chexpert_test_loader

def get_data_loaders_vindr(cfg):
    return get_data_loaders_structed(cfg, 'vindr')

def get_data_loaders_nih14(cfg):  
    return get_data_loaders_structed(cfg, 'nih14')

def get_data_loaders_padchest(cfg):    
    return get_data_loaders_structed(cfg, 'padchest')

def get_data_loaders_structed(cfg, dataset_name="None", mode = 'test'):    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = MultiSourceDataset(cfg, dataset_name=dataset_name, mode=mode, transform=transform)
    dataLoader = DataLoader(dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)

    return dataLoader
