import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
from PIL import Image

from .metrics import AUCProcessor

class CleanSingleDomainDataset(Dataset):
    """
    The Dataset class is designed to read only ONE single "clean" data domain.
    The application of noise will be handled externally.
    """
    def __init__(self, cfg, transform=None):
        base_domain_cfg = cfg.DATASET.BASE_DOMAIN
        self.root_dir = os.path.join(base_domain_cfg.PATH, base_domain_cfg.IMAGE_DIR)
        self.transform = transform
        self.labels_list = cfg.DATASET.LABELS_LIST
        
        csv_path = os.path.join(base_domain_cfg.PATH, base_domain_cfg.CSV)
        print(f"Loading clean data from: {csv_path}")
        self.df = pd.read_csv(csv_path)        
        self.image_col = 'image_id'
        print(f"Initialized clean dataset with {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row[self.image_col])
        
        if ("CheXpert-v1.0-small" in img_name):
            img_path = f"/home/ngoto/Working/Data/{img_name}"
        elif not os.path.isabs(img_name):
             img_path = os.path.join(self.root_dir, img_name)
        else:
             img_path = img_name
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: File not found at {img_path}. Returning None.")
            return None
            
        labels = torch.tensor(row[self.labels_list].values.astype('float'), dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return {'image': image, 'label': labels, 'domain': 'clean_base'}

def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return {'image': torch.empty(0), 'label': torch.empty(0), 'domain': []}
    return torch.utils.data.dataloader.default_collate(batch)


def build_loader_multilabel(cfg):
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    clean_dataset = CleanSingleDomainDataset(cfg, transform=base_transform)
    
    loader = DataLoader(
        clean_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,  # Important: DO NOT shuffle to ensure the dataset is traversed in order
        num_workers=cfg.LOADER.NUM_WORKS,
        collate_fn=collate_fn_skip_none,
        pin_memory=True
    )
    
    result_processor = AUCProcessor(num_classes=len(cfg.DATASET.LABELS_LIST))
    print("Clean data loader and AUC processor are built successfully.")
    return loader, result_processor