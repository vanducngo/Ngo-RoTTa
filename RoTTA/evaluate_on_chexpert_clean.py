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

# Import các thành phần từ project của bạn
from core.configs import cfg
from core.model import build_model
from Base.metrics import AUCProcessor
# KHÔNG CẦN import apply_corruption nữa
from core.utils import set_random_seed

# ==============================================================================
# Lớp Dataset để tải CheXpert SẠCH
# (Có thể tái sử dụng CleanSingleDomainDataset nếu bạn đã có)
# ==============================================================================

class CheXpertCleanDataset(Dataset):
    """Tải dữ liệu sạch từ tập test CheXpert."""
    def __init__(self, cfg):
        # Đọc cấu hình từ một mục cụ thể, ví dụ SOURCE_DOMAIN
        self.root_dir = cfg.DATASET.CHEXPERT_PATH
        csv_cfg_dir = cfg.DATASET.CHEXPERT_CSV
        csv_path = os.path.join(self.root_dir, csv_cfg_dir)

        self.labels_list = cfg.DATASET.LABELS_LIST
        
        print(f"Loading clean CheXpert data from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.image_col = 'image_id' # Tên cột trong file CSV của CheXpert
        print(f"Loaded {len(self.df)} clean samples from CheXpert.")

        # Định nghĩa transform, BAO GỒM cả Normalize
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row[self.image_col])
        img_path = os.path.join('/home/ngoto/Working/Data/', img_name)
        
        try:
            image_pil = Image.open(img_path).convert('RGB')
            # Áp dụng transform ở đây
            image_tensor = self.transform(image_pil)
        except FileNotFoundError:
            print(f"Warning: File not found at {img_path}. Returning None.")
            return None

        labels = torch.tensor(row[self.labels_list].values.astype('float'), dtype=torch.float32)
        
        # Dữ liệu trả về đã được transform và normalize đầy đủ
        return {'image': image_tensor, 'label': labels}

# (collate_fn không đổi)
def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return {'image': torch.empty(0), 'label': torch.empty(0)}
    return torch.utils.data.dataloader.default_collate(batch)

# ==============================================================================
# Hàm chính để chạy đánh giá
# ==============================================================================

def evaluate_zero_shot_on_chexpert_clean(cfg):
    """
    Tải mô hình nguồn và chạy đánh giá Zero-Shot trên CheXpert SẠCH.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Tải mô hình nguồn đã được fine-tuned
    print("Building and loading fine-tuned source model...")
    model = build_model(cfg)
    model.to(device)
    model.eval()

    # 2. Tạo DataLoader cho CheXpert SẠCH
    print("\nBuilding CheXpert Clean DataLoader...")
    chexpert_clean_dataset = CheXpertCleanDataset(cfg)
    chexpert_clean_loader = DataLoader(
        chexpert_clean_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.LOADER.NUM_WORKS,
        collate_fn=collate_fn_skip_none
    )
    
    # 3. Chuẩn bị processor
    processor = AUCProcessor(num_classes=len(cfg.DATASET.LABELS_LIST))

    # 4. Chạy đánh giá
    print("\nStarting Zero-Shot evaluation on CheXpert-Clean...")
    tbar = tqdm(chexpert_clean_loader, desc="Evaluating on CheXpert-Clean")
    with torch.no_grad():
        for data_package in tbar:
            if not data_package['image'].numel(): continue

            # Dữ liệu từ loader đã được transform và normalize
            images = data_package['image'].to(device)
            labels = data_package['label'].to(device)
            
            logits = model(images)
            probabilities = torch.sigmoid(logits)
            
            processor.process(probabilities.detach(), labels.detach(), ["chexpert-clean"] * len(labels))

    # 5. In kết quả
    processor.calculate()
    print("\n" + "="*50)
    print("ZERO-SHOT RESULTS ON CheXpert-Clean (Test Set)")
    print("="*50)
    print(processor.info())
    print("\n=> Use this Mean AUC as the new in-domain baseline.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script to evaluate a source model on the clean CheXpert test set.")
    parser.add_argument(
        '-cfg',
        '--config-file',
        metavar="FILE",
        required=True,
        help="Path to the main config file (must define SOURCE_DOMAIN for CheXpert test).",
        type=str)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    set_random_seed(cfg.SEED)
    
    evaluate_zero_shot_on_chexpert_clean(cfg)