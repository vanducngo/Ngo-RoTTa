import logging
import torch
import argparse
from sklearn.metrics import roc_auc_score
import numpy as np

from core.configs import cfg
from core.utils import *

# Import các thành phần cần thiết
from core.utils.constants import DEVICE
from core.utils.metrics import AUCProcessor 
from core.model import build_model
from core.data.multilabel_loader import build_loader_multilabel
from tqdm import tqdm
from setproctitle import setproctitle


def testZeroShotMultiLabel(cfg):
    logger = logging.getLogger("ZeroShot.test_time_multilabel")
    
    model = build_model(cfg)
    model.to(DEVICE)
    model.eval() # Đặt mô hình ở chế độ đánh giá

    logger.info("Model built and set to evaluation mode for zero-shot testing.")

    # Nạp dữ liệu từ các domain
    loader, processor = build_loader_multilabel(cfg)

    # --- THAY ĐỔI: Vòng lặp chỉ thực hiện inference, không adaptation ---
    tbar = tqdm(loader)
    # Tắt việc tính toán gradient để tăng tốc độ
    with torch.no_grad():
        for batch_id, data_package in enumerate(tbar):
            if not data_package['image'].numel():
                continue

            data, label, domain = data_package["image"], data_package['label'], data_package['domain']
            data, label = data.to(DEVICE), label.to(DEVICE)
            
            # Chỉ thực hiện một lượt forward
            logits = model(data)

            # Tính xác suất và đưa vào processor để tính AUC
            probabilities = torch.sigmoid(logits)
            processor.process(probabilities, label, domain)
            
            # Không cần hiển thị AUC tạm thời vì quá trình inference rất nhanh
            tbar.set_description("Running Zero-Shot Evaluation")
    
    # Tính toán kết quả cuối cùng
    processor.calculate()

    logger.info(f"--- Zero-Shot Results ---\n{processor.info()}")
    # In kết quả ra console để dễ theo dõi
    print(f"\n--- Zero-Shot Results ---\n{processor.info()}")


def main():
    parser = argparse.ArgumentParser("Pytorch Implementation for Zero-Shot Multi-Label Evaluation!")

    parser.add_argument(
        '-cfg',
        '--config-file',
        metavar="FILE",
        default="",
        help="path to the main config file (should contain dataset and model info)",
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

    # --- THAY ĐỔI: Chỉ cần nạp config, không cần freeze() nếu không TTA ---
    # Freeze() ngăn việc thay đổi config sau này, không cần thiết cho zero-shot
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze() # Có thể bỏ dòng này

    # --- THAY ĐỔI: Đặt tên tiến trình cho phù hợp ---
    ds = cfg.DATASET.NAME 
    setproctitle(f"ZeroShot-ML:{ds:>8s}")

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    # Tạo file log riêng cho kết quả zero-shot
    logger = setup_logger('ZeroShot-ML', cfg.OUTPUT_DIR, 0, filename='log_zero_shot.txt')
    logger.info(args)

    logger.info(f"Loaded configuration file: {args.config_file}")
    logger.info("Running with config:\n{}".format(cfg))

    set_random_seed(cfg.SEED)

    # Gọi hàm test zero-shot
    testZeroShotMultiLabel(cfg)


if __name__ == "__main__":
    main()