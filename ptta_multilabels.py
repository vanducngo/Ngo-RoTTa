import logging
import torch
import argparse
from sklearn.metrics import roc_auc_score
import numpy as np

from core.configs import cfg
from core.utils import *

# Import lớp processor mới
from core.utils.metrics import AUCProcessor 
from core.model import build_model
from core.data.multilabel_loader import build_loader_multilabel
from core.optim import build_optimizer
from core.adapter import build_adapter
from tqdm import tqdm
from setproctitle import setproctitle


def testTimeAdaptationMultiLabel(cfg):
    logger = logging.getLogger("TTA.test_time_multilabel")
    # model, optimizer
    model = build_model(cfg)

    optimizer = build_optimizer(cfg)

    tta_adapter = build_adapter(cfg)

    tta_model = tta_adapter(cfg, model, optimizer)
    tta_model.cpu()

    # Build_loader giờ sẽ tải dữ liệu từ các domain
    # và được thay thế bằng cấu hình về domain trong file config.
    # loader = build_loader_multilabel(cfg)
    loader, processor = build_loader_multilabel(cfg) 

    # Sử dụng AUCProcessor thay vì processor cũ
    # num_classes=len(cfg.DATASET.LABELS_LIST)
    # processor = AUCProcessor(num_classes=num_classes)

    tbar = tqdm(loader)
    for batch_id, data_package in enumerate(tbar):
        # Giả định data_package giờ chứa nhãn đa nhãn (multi-label)
        data, label, domain = data_package["image"], data_package['label'], data_package['domain']
        
        # Bỏ qua kiểm tra len(label) == 1 vì batch cuối vẫn xử lý được
        data, label = data.cpu(), label.cpu()
        
        # forward_and_adapt đã được sửa để xử lý đa nhãn
        logits = tta_model(data)

        # ### SỬA ĐỔI ###: Logic đánh giá chuyển sang tính AUC
        # Lấy xác suất từ logits bằng Sigmoid
        probabilities = torch.sigmoid(logits)
        
        # Đưa xác suất và nhãn thật vào processor để tính toán sau
        processor.process(probabilities, label, domain)
        
        if batch_id > 0 and batch_id % 10 == 0:
            # Gộp tất cả các batch đã thu thập lại
            temp_labels = np.concatenate(processor.all_labels, axis=0)
            temp_preds = np.concatenate(processor.all_predictions, axis=0)
            
            # Tính AUC cho từng lớp trên toàn bộ dữ liệu đã thu thập
            per_class_aucs = []
            for i in range(temp_labels.shape[1]): # Lặp qua từng lớp (bệnh)
                y_true_col = temp_labels[:, i]
                y_pred_col = temp_preds[:, i]
                
                # Chỉ tính AUC nếu có cả hai loại nhãn (0 và 1)
                if len(np.unique(y_true_col)) > 1:
                    try:
                        auc = roc_auc_score(y_true_col, y_pred_col)
                        per_class_aucs.append(auc)
                    except ValueError:
                        # Trường hợp hiếm gặp khác, bỏ qua
                        pass
            
            # Tính mean AUC từ các giá trị hợp lệ
            current_mean_auc = np.mean(per_class_aucs) if per_class_aucs else 0.0

            # Cập nhật thanh tiến trình
            if hasattr(tta_model, "mem"):
                tbar.set_postfix(m_auc=f"{current_mean_auc:.3f}", bank=tta_model.mem.get_occupancy())
            else:
                tbar.set_postfix(m_auc=f"{current_mean_auc:.3f}")

    
    # ### SỬA ĐỔI ###: Tính toán kết quả cuối cùng
    processor.calculate()

    logger.info(f"All Results\n{processor.info()}")
    print(f"RoTTa Results\n{processor.info()}")

def main():
    # Phần main() để parse config không cần thay đổi nhiều
    parser = argparse.ArgumentParser("Pytorch Implementation for Multi-Label Test Time Adaptation!")

    parser.add_argument(
        '-cfg',
        '--config-file',
        metavar="FILE",
        default="",
        help="path to the main config file",
        type=str)
    
    # parser.add_argument(
    #     '-acfg',
    #     '--adapter-config-file',
    #     metavar="FILE",
    #     default="",
    #     help="path to adapter config file",
    #     type=str)
    # parser.add_argument(
    #     '-dcfg',
    #     '--dataset-config-file',
    #     metavar="FILE",
    #     help="path to dataset config file for multi-domain/multi-label setup",
    #     type=str)
    
    # Bỏ order-config-file nếu không cần thiết cho kịch bản domain
    # parser.add_argument(
    #     '-ocfg',
    #     '--order-config-file',
    #     ...
    # )

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
    # cfg.merge_from_file(args.adapter_config_file)
    # cfg.merge_from_file(args.dataset_config_file)
    # if not args.order_config_file == "":
    #     cfg.merge_from_file(args.order_config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    ds = cfg.DATASET.NAME # ### SỬA ĐỔI ###: Lấy tên dataset từ config mới
    adapter = cfg.ADAPTER.NAME
    setproctitle(f"TTA-ML:{ds:>8s}:{adapter:<10s}")

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger('TTA-ML', cfg.OUTPUT_DIR, 0, filename=cfg.LOG_DEST)
    logger.info(args)

    # logger.info(f"Loaded configuration file: \n"
    #             f"\tadapter: {args.adapter_config_file}\n"
    #             f"\tdataset: {args.dataset_config_file}\n")
    #             # f"\torder: {args.order_config_file}")
    logger.info("Running with config:\n{}".format(cfg))

    set_random_seed(cfg.SEED)

    # Gọi hàm mới
    testTimeAdaptationMultiLabel(cfg)


if __name__ == "__main__":
    main()