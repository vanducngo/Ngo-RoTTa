import logging
import torch
import argparse
from sklearn.metrics import roc_auc_score
import numpy as np

from core.configs import cfg
from core.data.corruptions import apply_corruption
from core.utils import *

# Import lớp processor mới
from core.utils.constants import IS_CPU_DEVICE
from core.utils.metrics import AUCProcessor 
from core.model import build_model
from core.data.multilabel_loader import build_loader_multilabel
from core.optim import build_optimizer
from core.adapter import build_adapter
from tqdm import tqdm
from setproctitle import setproctitle


def testTimeAdaptationMultiLabel(cfg):
    logger = logging.getLogger("TTA.test_time_multilabel")
    
    # --- Khởi tạo Model và Adapter (không đổi) ---
    model = build_model(cfg)
    optimizer = build_optimizer(cfg)
    tta_adapter = build_adapter(cfg)
    tta_model = tta_adapter(cfg, model, optimizer)

    if IS_CPU_DEVICE:
        tta_model.cpu()
        tta_model.model_ema.cpu()
    else:
        tta_model.cuda()
        tta_model.model_ema.cuda()
    
    # --- Logic điều khiển chế độ thích ứng ---
    adaptation_mode = cfg.DATASET.ADAPTATION_MODE

    if adaptation_mode == 'corruption':
        # --- Chạy chế độ ĐA-NHIỄU ---
        logger.info("Running in 'corruption' adaptation mode.")
        
        # Lấy danh sách nhiễu và Dataloader cho domain cơ sở
        corruptions_list = cfg.DATASET.TEST_CORRUPTIONS if cfg.DATASET.TEST_CORRUPTIONS else ['none']
        severity = cfg.DATASET.SEVERITY
        corruption_idx = 0
        loader, processor = build_loader_multilabel(cfg)

        tbar = tqdm(loader, desc="Adapt on Corruptions")
        for batch_id, data_package in enumerate(tbar):
            if not data_package['image'].numel(): continue

            clean_images, labels = data_package["image"], data_package['label']

            clean_images, labels = clean_images.cuda(), labels.cuda()
            current_corruption = corruptions_list[corruption_idx]            
            # Áp dụng nhiễu
            data_to_adapt = apply_corruption(clean_images, current_corruption, severity)
            data_to_adapt = data_to_adapt.cuda()
            
            # Thực hiện TTA
            logits = tta_model(data_to_adapt)
            
            # Xử lý kết quả (giữ nguyên)
            probabilities = torch.sigmoid(logits)
            domain_info = [current_corruption] * data_to_adapt.size(0)
            processor.process(probabilities, labels, domain_info)
            
            if batch_id > 0 and batch_id % 10 == 0:
                # Tính AUC tạm thời trên dữ liệu đã thu thập
                temp_labels = np.concatenate(processor.all_labels, axis=0)
                temp_preds = np.concatenate(processor.all_predictions, axis=0)
                valid_aucs = [roc_auc_score(temp_labels[:, i], temp_preds[:, i]) 
                              for i in range(temp_labels.shape[1]) 
                              if len(np.unique(temp_labels[:, i])) > 1]
                current_mean_auc = np.mean(valid_aucs) if valid_aucs else 0.0

                if hasattr(tta_model, "mem"):
                    tbar.set_postfix(m_auc=f"{current_mean_auc:.3f}", bank=tta_model.mem.get_occupancy(), severity={severity})
                else:
                    tbar.set_postfix(m_auc=f"{current_mean_auc:.3f}")

            corruption_idx = (corruption_idx + 1) % len(corruptions_list)

    elif adaptation_mode == 'domain':
        # --- Chạy chế độ ĐA-DOMAIN ---
        logger.info("Running in 'domain' adaptation mode.")
        
        # Lặp qua từng domain được định nghĩa trong config
        test_domains = cfg.DATASET.TEST_DOMAINS
        
        # Dùng một processor duy nhất để tổng hợp kết quả qua tất cả các domain
        processor = AUCProcessor(num_classes=len(cfg.DATASET.LABELS_LIST))

        for domain_name in test_domains:
            logger.info(f"--- Starting adaptation on domain: {domain_name} ---")
            
            # Tạo một config tạm thời để build_loader chỉ cho domain hiện tại
            domain_cfg = cfg.clone()
            # Đặt domain hiện tại làm BASE_DOMAIN để loader có thể đọc
            domain_cfg.defrost()
            domain_cfg.DATASET.BASE_DOMAIN.PATH = getattr(cfg.DATASET, f"{domain_name.upper()}_PATH")
            domain_cfg.DATASET.BASE_DOMAIN.CSV = getattr(cfg.DATASET, f"{domain_name.upper()}_CSV")
            domain_cfg.DATASET.BASE_DOMAIN.IMAGE_DIR = getattr(cfg.DATASET, f"{domain_name.upper()}_IMAGE_DIR")
            domain_cfg.freeze()

            # Build loader chỉ cho domain này
            loader, _ = build_loader_multilabel(domain_cfg)

            tbar = tqdm(loader, desc=f"Adapt on {domain_name}")
            for batch_id, data_package in enumerate(tbar):
                if not data_package['image'].numel(): continue
                
                # Dữ liệu đã là của domain này, không cần áp dụng thêm gì
                data_to_adapt, labels = data_package["image"].cuda(), data_package['label'].cuda()
                
                # Thực hiện TTA
                logits = tta_model(data_to_adapt)

                # Xử lý kết quả
                probabilities = torch.sigmoid(logits)
                domain_info = [domain_name] * data_to_adapt.size(0)
                processor.process(probabilities, labels, domain_info)
                
                if batch_id > 0 and batch_id % 10 == 0:
                    # Tính AUC tạm thời trên TOÀN BỘ dữ liệu đã thu thập (bao gồm cả các domain trước)
                    temp_labels = np.concatenate(processor.all_labels, axis=0)
                    temp_preds = np.concatenate(processor.all_predictions, axis=0)
                    valid_aucs = [roc_auc_score(temp_labels[:, i], temp_preds[:, i]) 
                                  for i in range(temp_labels.shape[1]) 
                                  if len(np.unique(temp_labels[:, i])) > 1]
                    current_mean_auc = np.mean(valid_aucs) if valid_aucs else 0.0

                    if hasattr(tta_model, "mem"):
                        tbar.set_postfix(m_auc=f"{current_mean_auc:.3f}", bank=tta_model.mem.get_occupancy())
                    else:
                        tbar.set_postfix(m_auc=f"{current_mean_auc:.3f}")
            
            # Quan trọng: Không reset model giữa các domain để mô phỏng continual adaptation
    
    else:
        raise ValueError(f"Unknown ADAPTATION_MODE: {adaptation_mode}")

    # --- Tính toán và in kết quả cuối cùng (chung cho cả hai chế độ) ---
    logger.info(f"--- Final Results for mode '{adaptation_mode}' ---\n{processor.info()}")
    print(f"--- Final Results for mode '{adaptation_mode}' ---\n{processor.info()}")

def main():
    # Phần main() để parse config 
    parser = argparse.ArgumentParser("Pytorch Implementation for Multi-Label Test Time Adaptation!")

    parser.add_argument(
        '-cfg',
        '--config-file',
        metavar="FILE",
        default="",
        help="path to the main config file",
        type=str)

    parser.add_argument(
        'opts',
        help='modify the configuration by command line',
        nargs=argparse.REMAINDER,
        default=None)

    args = parser.parse_args()

    if len(args.opts) > 0:
        args.opts[-1] = args.opts[-1].strip('\r\n')

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    ds = cfg.DATASET.NAME
    adapter = cfg.ADAPTER.NAME
    setproctitle(f"TTA-ML:{ds:>8s}:{adapter:<10s}")

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger('TTA-ML', cfg.OUTPUT_DIR, 0, filename=cfg.LOG_DEST)
    logger.info(args)

    logger.info("Running with config:\n{}".format(cfg))

    # set_random_seed(cfg.SEED)
    set_random_seed(42)

    # Gọi hàm mới
    testTimeAdaptationMultiLabel(cfg)


if __name__ == "__main__":
    main()