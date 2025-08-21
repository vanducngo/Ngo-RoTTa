import os
import sys
# Lấy đường dẫn của thư mục gốc project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import argparse
import numpy as np

# Import các thành phần từ project của bạn
from core.configs import cfg
from core.model import build_model
from Base.metrics import AUCProcessor
from Base.corruptions import apply_corruption
from core.utils import set_random_seed

# ==============================================================================
# Lớp Dataset để tạo "CheXpert-C" on-the-fly
# ==============================================================================

class CheXpertCorruptionDataset(Dataset):
    """
    Tạo một Dataset lớn chứa dữ liệu từ tập test CheXpert sạch,
    được sắp xếp và làm nhiễu theo từng loại, mô phỏng CIFAR-C.
    """
    def __init__(self, cfg):
        # 1. Đọc cấu hình
        self.root_dir = cfg.DATASET.CHEXPERT_PATH
        csv_cfg_dir = cfg.DATASET.CHEXPERT_CSV
        csv_path = os.path.join(self.root_dir, csv_cfg_dir)

        self.corruptions_to_apply = cfg.DATASET.TEST_CORRUPTIONS
        self.severity = cfg.DATASET.SEVERITY
        self.labels_list = cfg.DATASET.LABELS_LIST

        # 2. Tải dữ liệu CheXpert "sạch"
        print(f"Loading base CheXpert data from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.image_col = 'image_id' # Giả sử cho CheXpert
        print(f"Loaded {len(self.df)} base samples from CheXpert.")

        # 3. Định nghĩa các transform cơ bản (KHÔNG Normalize)
        self.to_tensor_transform = transforms.ToTensor()
        self.resize_transform = transforms.Resize((224, 224))

        # 4. Tạo danh sách mẫu lớn, được sắp xếp theo từng loại nhiễu
        self.all_samples = []
        print("Building CheXpert-C stream...")
        for corruption_name in self.corruptions_to_apply:
            for i in range(len(self.df)):
                self.all_samples.append((i, corruption_name))
                
        print(f"CheXpert-C Dataset created with {len(self.all_samples)} total samples.")

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        sample_idx, corruption_name = self.all_samples[idx]
        row = self.df.iloc[sample_idx]
        img_name = str(row[self.image_col])
        img_path = os.path.join('/home/ngoto/Working/Data/', img_name)
        
        try:
            image_pil = self.resize_transform(Image.open(img_path).convert('RGB'))
        except FileNotFoundError as e:
            print(f'fine not found error: {e}')
            return None

        image_tensor = self.to_tensor_transform(image_pil)
        corrupted_tensor = apply_corruption(image_tensor, corruption_name, self.severity)
        
        labels = torch.tensor(row[self.labels_list].values.astype('float'), dtype=torch.float32)
        
        return {'image': corrupted_tensor, 'label': labels, 'domain': corruption_name}

# (collate_fn không đổi)
def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return {'image': torch.empty(0), 'label': torch.empty(0)}
    return torch.utils.data.dataloader.default_collate(batch)

# ==============================================================================
# Hàm chính để chạy đánh giá
# ==============================================================================

def evaluate_zero_shot_on_chexpert_c(cfg):
    """
    Tải mô hình nguồn, tạo CheXpert-C, và chạy đánh giá Zero-Shot.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Tải mô hình nguồn đã được fine-tuned
    print("Building and loading fine-tuned source model...")
    model = build_model(cfg)
    model.to(device)
    model.eval()

    # 2. Tạo DataLoader cho CheXpert-C
    print("\nBuilding CheXpert-C DataLoader...")
    chexpert_c_dataset = CheXpertCorruptionDataset(cfg)
    chexpert_c_loader = DataLoader(
        chexpert_c_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.LOADER.NUM_WORKS,
        collate_fn=collate_fn_skip_none
    )
    
    # 3. Chuẩn bị processor và transform để Normalize
    processor = AUCProcessor(num_classes=len(cfg.DATASET.LABELS_LIST), class_names=cfg.DATASET.LABELS_LIST)
    normalize_transform = transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    # 4. Chạy đánh giá
    print("\nStarting Zero-Shot evaluation on CheXpert-C...")
    tbar = tqdm(chexpert_c_loader, desc="Evaluating on CheXpert-C")
    with torch.no_grad():
        for data_package in tbar:
            if not data_package['image'].numel(): continue

            images_unnormalized = data_package['image'].to(device)
            labels = data_package['label'].to(device)
            # Lấy thông tin domain (tên nhiễu) từ data_package
            domains = data_package['domain'] 
            
            images = normalize_transform(images_unnormalized)
            logits = model(images)
            probabilities = torch.sigmoid(logits)
            
            # Truyền danh sách domains vào processor
            processor.process(probabilities.detach(), labels.detach(), domains)

    # In kết quả
    processor.calculate()
    print("\n" + "="*50)
    print("ZERO-SHOT RESULTS ON CheXpert-C (Per Corruption)")
    print("="*50)
    print(processor.info())


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script to evaluate a source model on a generated CheXpert-C dataset.")
    parser.add_argument(
        '-cfg',
        '--config-file',
        metavar="FILE",
        required=True,
        help="Path to the main config file.",
        type=str)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    set_random_seed(cfg.SEED)
    
    evaluate_zero_shot_on_chexpert_c(cfg)