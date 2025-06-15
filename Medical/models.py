import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(cfg):
    """
    Tải mô hình ResNet18 và điều chỉnh cho bài toán.
    Đầu ra là logits, không phải xác suất.
    """
    print(">>> Loading model...")
    
    # Sử dụng ResNet18 vì nó đang cho kết quả tốt hơn
    if cfg.MODEL.ARCH == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1 # Dùng V1 ổn định hơn
        model = resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, cfg.MODEL.NUM_CLASSES)
    # Thêm các mô hình khác nếu muốn
    # elif cfg.MODEL.ARCH == 'resnet50':
    #     ...

    print(f"Model {cfg.MODEL.ARCH} pre-trained on ImageNet loaded.")
    print(f"Model adapted for {cfg.MODEL.NUM_CLASSES} classes (outputting logits).")
    
    return model

if __name__ == '__main__':
    # Test thử chức năng của file
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('configs/base_config.yaml')
    model = get_model(cfg)
    print(model)
    # Thử forward một ảnh giả
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print("Dummy output shape:", output.shape)