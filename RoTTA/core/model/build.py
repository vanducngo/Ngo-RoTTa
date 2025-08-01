from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from Core.models import get_pretrained_model

def build_model(cfg):
    """
    Xây dựng mô hình.
    - Nếu là dataset gốc (cifar10, cifar100), tải mô hình đã huấn luyện từ robustbench.
    - Nếu là dataset đa nhãn (như CXR), tải một mô hình nền và thay thế lớp classifier.
    """
    dataset_name = cfg.DATASET.NAME

    if dataset_name in ["cifar10", "cifar100"]:
        # --- Logic gốc cho CIFAR ---
        print(f"Building pre-trained model for {dataset_name}...")
        base_model = load_model(
            cfg.MODEL.ARCH, 
            cfg.CKPT_DIR,
            dataset_name, 
            ThreatModel.corruptions
        ).cpu()

    elif 'CXR' in dataset_name:
        base_model = get_pretrained_model(cfg)
    else:
        raise NotImplementedError(f"Model building logic not implemented for dataset: {dataset_name}")

    return base_model