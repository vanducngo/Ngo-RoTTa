import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import WeightedRandomSampler

# Giả định data_mapping.py đã được import
from data_mapping import FINAL_LABEL_SET, map_chexpert_labels, map_vindr_labels

# ... (hàm read_dicom_image giữ nguyên) ...
def read_dicom_image(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.read_file(path)
    if voi_lut: data = apply_voi_lut(dicom.pixel_array, dicom)
    else: data = dicom.pixel_array
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1": data = np.amax(data) - data
    data = data - np.min(data)
    data = data / (np.max(data) + 1e-6)
    data = (data * 255).astype(np.uint8)
    return Image.fromarray(data).convert('RGB')

class MultiSourceDataset(Dataset):
    # ... (Nội dung lớp này giữ nguyên như bạn đã viết, chỉ cần đảm bảo nó trả về df đã map) ...
    def __init__(self, cfg, dataset_name, mode='train', transform=None):
        # ... logic của bạn ở đây ...
        # Ví dụ cho CheXpert
        if dataset_name == 'chexpert':
            # Nếu mode là train hoặc valid, dùng file train.csv
            csv_file = cfg.DATA.CHEXPERT_TRAIN_CSV if mode in ['train', 'valid'] else cfg.DATA.CHEXPERT_TEST_CSV
            csv_path = os.path.join(cfg.DATA.CHEXPERT_PATH, csv_file)
            raw_df = pd.read_csv(csv_path)
            self.df = map_chexpert_labels(raw_df)
            self.root_dir = cfg.DATA.CHEXPERT_PATH
            self.image_col = 'Path'
            self.path_prefix = ''
            self.is_dicom = False
        elif dataset_name == 'vindr':
            csv_path = os.path.join(cfg.DATA.VINDR_PATH, cfg.DATA.VINDR_EVAL_CSV)
            raw_df = pd.read_csv(csv_path)
            self.df = map_vindr_labels(raw_df)
            self.root_dir = os.path.join(cfg.DATA.VINDR_PATH, cfg.DATA.VINDR_IMAGE_DIR)
            self.image_col = 'image_id'
            self.path_prefix = '.dicom'
            self.is_dicom = True
        # ...
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        # ... logic của bạn ở đây ...
        img_name = self.df.iloc[idx][self.image_col]
        if self.dataset_name == 'chexpert': img_path = os.path.join(self.root_dir, img_name)
        else: img_path = os.path.join(self.root_dir, img_name + self.path_prefix)
        try:
            if self.is_dicom: image = read_dicom_image(img_path)
            else: image = Image.open(img_path).convert('RGB')
        except Exception: return torch.empty(0), torch.empty(0)
        labels = self.df.iloc[idx][FINAL_LABEL_SET].values.astype('float')
        labels = torch.tensor(labels, dtype=torch.float32)
        if self.transform: image = self.transform(image)
        return image, labels


def collate_fn(batch):
    # ... (giữ nguyên) ...
    batch = list(filter(lambda x: x[0].numel() > 0, batch))
    if not batch: return torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)

def get_data_loaders(cfg):
    """Tạo DataLoader cho huấn luyện, xác thực, và kiểm tra."""
    
    # Định nghĩa các phép tăng cường dữ liệu
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform cho valid/test không có augmentation ngẫu nhiên
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Tải toàn bộ tập train của CheXpert
    full_train_dataset = MultiSourceDataset(cfg, dataset_name='chexpert', mode='train', transform=None)
    
    # Chia chỉ số ra thành train và valid
    indices = list(range(len(full_train_dataset)))
    train_idx, valid_idx = train_test_split(
        indices,
        test_size=cfg.TRAINING.VALIDATION_SPLIT,
        random_state=42 # Đảm bảo chia giống nhau mỗi lần chạy
    )
    
    # Tạo các đối tượng Subset với transform tương ứng
    train_dataset = Subset(full_train_dataset, train_idx)
    train_dataset.dataset.transform = train_transform # Gán transform cho tập train
    valid_dataset = Subset(full_train_dataset, valid_idx)
    valid_dataset.dataset.transform = eval_transform # Gán transform cho tập valid

    # Tạo DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAINING.BATCH_SIZE,
        shuffle=True, # Shuffle khi không dùng sampler
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    if cfg.TRAINING.USE_WEIGHTED_SAMPLER:
        print(">>> Using WeightedRandomSampler for training...")
        # Lấy nhãn của các mẫu trong tập train
        labels_for_sampler = np.array([full_train_dataset.df.iloc[i][FINAL_LABEL_SET].values for i in train_idx])
        class_counts = np.sum(labels_for_sampler, axis=0)
        class_weights = 1.0 / (class_counts + 1e-6)
        
        # Tính trọng số cho mỗi mẫu
        sample_weights = np.sum(labels_for_sampler * class_weights, axis=1)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        
        # Tạo lại train_loader với sampler, không shuffle
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.TRAINING.BATCH_SIZE,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.TRAINING.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Các loader cho tập test
    chexpert_test_dataset = MultiSourceDataset(cfg, dataset_name='chexpert', mode='test', transform=eval_transform)
    chexpert_test_loader = DataLoader(chexpert_test_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    vindr_test_dataset = MultiSourceDataset(cfg, dataset_name='vindr', mode='test', transform=eval_transform)
    vindr_test_loader = DataLoader(vindr_test_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, valid_loader, chexpert_test_loader, vindr_test_loader