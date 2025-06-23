import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import warnings

# Giả định file data_mapping.py tồn tại và chứa các biến/hàm sau:
# COMMON_DISEASES: list[str]
# map_chexpert_labels: function
# map_vindr_labels: function
# map_chestxray14_labels: function
from data_mapping import COMMON_DISEASES, map_chexpert_labels, map_vindr_labels, map_chestxray14_labels

def read_dicom_image(path, voi_lut=True, fix_monochrome=True):
    """
    Đọc ảnh từ file DICOM và chuyển thành định dạng mà PyTorch có thể xử lý.
    """
    dicom = pydicom.read_file(path)
    
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
        
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    
    image = Image.fromarray(data).convert('RGB')
    return image

class MultiSourceDataset(Dataset):
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
            # self.df = pd.read_csv(csv_path)
            self.root_dir = cfg.DATA.CHEXPERT_PATH
            self.image_col = 'Path'
            self.path_prefix = '' # CheXpert đã có đường dẫn đầy đủ
            self.is_dicom = False
            
        elif self.dataset_name == 'vindr':
            image_dir_name = cfg.DATA.VINDR_IMAGE_DIR
            csv_file_name = cfg.DATA.VINDR_EVAL_CSV

            csv_path = os.path.join(cfg.DATA.VINDR_PATH, csv_file_name)
            raw_df = pd.read_csv(csv_path)
            # Ánh xạ nhãn từ file gốc của VinDr về các nhãn chung
            self.df = map_vindr_labels(raw_df)
            
            # Thư mục gốc chứa ảnh dicom
            self.root_dir = os.path.join(cfg.DATA.VINDR_PATH, image_dir_name)
            self.image_col = 'image_id'
            self.path_prefix = '.dicom' # Thêm đuôi file cho ảnh
            self.is_dicom = True # Đánh dấu đây là dữ liệu DICOM
            
        elif self.dataset_name == 'chestxray14':
            csv_path = os.path.join(cfg.DATA.CHESTXRAY14_PATH, cfg.DATA.CHESTXRAY14_CSV)
            raw_df = pd.read_csv(csv_path)
            self.df = map_chestxray14_labels(raw_df)
            self.root_dir = os.path.join(cfg.DATA.CHESTXRAY14_PATH, 'images') 
            self.image_col = 'image_id'
            self.path_prefix = '' # ChestXray14 thường là .png và đã có trong tên file
            self.is_dicom = False
            
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
        print(f"Loaded and mapped {self.dataset_name} ({mode}) with {len(self.df)} samples.")
        print(f"Common diseases being used: {COMMON_DISEASES}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Lấy tên file/đường dẫn ảnh từ dataframe
        img_name = self.df.iloc[idx][self.image_col]
        
        # Tạo đường dẫn đầy đủ
        if self.dataset_name == 'chexpert':
            # CheXpert có cấu trúc đường dẫn đặc biệt
            img_path = os.path.join(self.root_dir, img_name)
        else:
            img_path = os.path.join(self.root_dir, img_name + self.path_prefix)
            
        try:
            # Đọc ảnh dựa trên loại file
            if self.is_dicom:
                image = read_dicom_image(img_path)
            else:
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
            
        # print('VanDucNgo', labels)
        return image, labels

# Hàm helper để bỏ qua các mẫu bị lỗi
def collate_fn(batch):
    batch = list(filter(lambda x: x[0].numel() > 0, batch))
    if not batch:
        return torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)

def get_data_loaders(cfg):
    """Tạo các DataLoader cho việc huấn luyện và kiểm tra."""
    
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
    
    # Nguồn: CheXpert
    train_dataset = MultiSourceDataset(cfg, dataset_name='chexpert', mode='train', transform=transform)
    # Lấy subset để train nhanh hơn
    train_subset = torch.utils.data.Subset(train_dataset, range(len(train_dataset)))
    train_loader = DataLoader(train_subset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    chexpert_test_dataset = MultiSourceDataset(cfg, dataset_name='chexpert', mode='test', transform=transform)
    chexpert_test_loader = DataLoader(chexpert_test_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Đích 1: VinDr-CXR (đánh giá trên tập test)
    # Sử dụng mode='test' để tự động chọn file test từ config
    vindr_test_dataset = MultiSourceDataset(cfg, dataset_name='vindr', mode='test', transform=transform)
    vindr_test_loader = DataLoader(vindr_test_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    # vindr_test_loader = None
    
    # Đích 2: ChestX-ray14
    # (Bạn có thể thêm vào đây nếu muốn kiểm tra trên cả 3)
    # chestxray14_test_dataset = ...
    # chestxray14_test_loader = ...

    return train_loader, chexpert_test_loader, vindr_test_loader