import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(cfg):
    """
    Tải mô hình ResNet50 và điều chỉnh cho bài toán phân loại đa nhãn.
    """
    print(">>> Loading model...")
    
    # Bước 1: Tải pre-trained model trên ImageNet.
    # Lưu ý: Pre-trained trên ImageNet (1000 lớp ảnh tự nhiên) thường hiệu quả hơn
    # cho ảnh y tế so với CIFAR-100 (100 lớp ảnh nhỏ). 
    # Nếu bạn vẫn muốn CIFAR-100, bạn cần tìm một checkpoint tương ứng.
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    
    print(f"Model pre-trained on ImageNet loaded.")

    # Bước 2: Thay thế lớp phân loại cuối cùng (fully connected layer)
    # để phù hợp với số lớp bệnh của chúng ta.
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5), # Tỷ lệ dropout, 0.5 là giá trị phổ biến
        nn.Linear(num_ftrs, cfg.MODEL.NUM_CLASSES),
        nn.Sigmoid() # Dùng Sigmoid cho bài toán đa nhãn (multi-label)
    )

    print(f"Model adapted for {cfg.MODEL.NUM_CLASSES} classes with Sigmoid activation.")
    
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