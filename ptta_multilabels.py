import logging
import torch
import argparse
from sklearn.metrics import roc_auc_score
import numpy as np
import copy
import os

from core.configs import cfg
from core.utils import *

# Import các thành phần cần thiết
from core.utils.constants import IS_CPU_DEVICE
from core.utils.metrics import AUCProcessor 
from core.model import build_model
from core.data.multilabel_loader import build_loader_multilabel
from core.data.corruptions import apply_corruption
from core.optim import build_optimizer
from core.adapter import build_adapter
from tqdm import tqdm
from setproctitle import setproctitle


def testTimeAdaptationMultiLabel(cfg):
    """
    Hàm chính thực hiện Test-Time Adaptation cho bài toán đa nhãn.
    Hỗ trợ hai chế độ: 'corruption' và 'domain', và có thể chạy nhiều epoch.
    """
    logger = logging.getLogger("TTA.test_time_multilabel")
    
    # --- 1. Khởi tạo Model, Adapter, Optimizer (chỉ một lần) ---
    model = build_model(cfg)
    optimizer = build_optimizer(cfg)
    tta_adapter = build_adapter(cfg)
    tta_model = tta_adapter(cfg, model, optimizer)

    if not IS_CPU_DEVICE:
        tta_model.cuda()
        if hasattr(tta_model, 'model_ema'):
            tta_model.model_ema.cuda()
    
    logger.info("Model and TTA adapter initialized.")

    # --- 2. Vòng lặp Epochs ---
    num_epochs = 1
    best_auc = 0.0
    best_model_state = None
    adaptation_mode = cfg.DATASET.ADAPTATION_MODE

    for epoch in range(1, num_epochs + 1):
        logger.info(f"=========== Starting TTA Epoch {epoch}/{num_epochs} ===========")

        # 2.1 Tạo lại processor cho mỗi epoch để đo lường riêng lẻ
        processor = AUCProcessor(num_classes=len(cfg.DATASET.LABELS_LIST))

        if adaptation_mode == 'corruption':
            # --- Chế độ ĐA-NHIỄU ---
            logger.info("Running in 'corruption' adaptation mode for this epoch.")
            
            corruptions_list = cfg.DATASET.TEST_CORRUPTIONS if cfg.DATASET.TEST_CORRUPTIONS else ['none']
            severity = cfg.DATASET.SEVERITY
            corruption_idx = 0
            loader, _ = build_loader_multilabel(cfg)

            tbar = tqdm(loader, desc=f"Epoch {epoch} [Corruption]")
            for batch_id, data_package in enumerate(tbar):
                if not data_package['image'].numel(): continue

                clean_images, labels = data_package["image"].cuda(), data_package['label'].cuda()
                current_corruption = corruptions_list[corruption_idx]
                data_to_adapt = apply_corruption(clean_images, current_corruption, severity)
                
                logits = tta_model(data_to_adapt)
                
                probabilities = torch.sigmoid(logits)
                domain_info = [current_corruption] * data_to_adapt.size(0)
                processor.process(probabilities, labels, domain_info)
                
                if batch_id > 0 and batch_id % 50 == 0:
                    if len(processor.all_labels) > 0:
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
                
                corruption_idx = (corruption_idx + 1) % len(corruptions_list)

        elif adaptation_mode == 'domain':
            # --- Chế độ ĐA-DOMAIN ---
            logger.info("Running in 'domain' adaptation mode for this epoch.")
            
            test_domains = cfg.DATASET.TEST_DOMAINS
            
            for domain_name in test_domains:
                logger.info(f"--- Adapting on domain: {domain_name} (Epoch {epoch}) ---")
                
                domain_cfg = cfg.clone(); domain_cfg.defrost()
                domain_cfg.DATASET.BASE_DOMAIN.PATH = getattr(cfg.DATASET, f"{domain_name.upper()}_PATH")
                domain_cfg.DATASET.BASE_DOMAIN.CSV = getattr(cfg.DATASET, f"{domain_name.upper()}_CSV")
                domain_cfg.DATASET.BASE_DOMAIN.IMAGE_DIR = getattr(cfg.DATASET, f"{domain_name.upper()}_IMAGE_DIR", "")
                domain_cfg.freeze()
                loader, _ = build_loader_multilabel(domain_cfg)

                tbar = tqdm(loader, desc=f"Epoch {epoch} [{domain_name}]")
                for batch_id, data_package in enumerate(tbar):
                    if not data_package['image'].numel(): continue
                    
                    data_to_adapt, labels = data_package["image"].cuda(), data_package['label'].cuda()
                    logits = tta_model(data_to_adapt)
                    
                    probabilities = torch.sigmoid(logits)
                    domain_info = [domain_name] * data_to_adapt.size(0)
                    processor.process(probabilities, labels, domain_info)
                    
                    if batch_id > 0 and batch_id % 10 == 0:
                        if len(processor.all_labels) > 0:
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
        else:
            raise ValueError(f"Unknown ADAPTATION_MODE: '{adaptation_mode}'. Must be 'corruption' or 'domain'.")

        # 2.2 Kết thúc một epoch, tính toán và ghi lại kết quả
        processor.calculate()
        epoch_auc = processor.results.get('mean_auc', 0.0)
        
        logger.info(f"--- Epoch {epoch} Final Results ---\n{processor.info()}")
        print(f"\n--- Epoch {epoch} Final Results ---\n{processor.info()}")

        # 2.3 Lưu lại mô hình tốt nhất dựa trên AUC của epoch
        if epoch_auc > best_auc:
            best_auc = epoch_auc
            best_model_state = {
                'student': copy.deepcopy(tta_model.model.state_dict()),
                'teacher': copy.deepcopy(tta_model.model_ema.state_dict()) if hasattr(tta_model, 'model_ema') else None
            }
            logger.info(f"New best AUC found: {best_auc:.4f}. Saving model state for epoch {epoch}.")

    # --- 3. Kết thúc tất cả các epoch ---
    logger.info(f"=========== TTA Finished ===========")
    logger.info(f"Best Mean AUC achieved across {num_epochs} epochs: {best_auc:.4f}")
    
    if best_model_state and cfg.OUTPUT_DIR:
        save_path = os.path.join(cfg.OUTPUT_DIR, "best_tta_model.pth")
        torch.save(best_model_state, save_path)
        logger.info(f"Best model state saved to {save_path}")

def main():
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
    
    # Thêm các khóa mặc định để tránh lỗi
    cfg.defrost()
    cfg.TEST.NUM_EPOCHS = cfg.TEST.get("NUM_EPOCHS", 1)
    cfg.freeze()

    ds = cfg.DATASET.NAME
    adapter = cfg.ADAPTER.NAME
    setproctitle(f"TTA-ML:{ds:>8s}:{adapter:<10s}")

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger('TTA-ML', cfg.OUTPUT_DIR, 0, filename=cfg.LOG_DEST)
    logger.info(args)
    logger.info(f"Loaded configuration file: {args.config_file}")
    logger.info("Running with config:\n{}".format(cfg))

    set_random_seed(cfg.SEED)
    
    testTimeAdaptationMultiLabel(cfg)


if __name__ == "__main__":
    main()