from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from Base.models import get_pretrained_model

def build_model(cfg):
    """
    Build the model.
    - If it's a standard dataset (e.g., CIFAR-10, CIFAR-100), load a pretrained model from RobustBench.
    - multi-label dataset (e.g., CXR), load a backbone model and replace the classifier layer.
    """
    dataset_name = cfg.DATASET.NAME

    if dataset_name in ["cifar10", "cifar100"]:
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