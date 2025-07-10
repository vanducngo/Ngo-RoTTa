import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights, densenet121, DenseNet121_Weights

def set_parameter_requires_grad(model, feature_extracting):
    """
    Hàm helper để đóng băng các tham số.
    Nếu feature_extracting = True, tất cả các tham số sẽ bị đóng băng.
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model_chexpert_14(cfg):
    return get_model(cfg, feature_extract=False, useWeight = True, numclasses=5)


# def get_model(cfg, feature_extract=False, useWeight = True, numclasses = 5):
#     """
#     Tải mô hình và tùy chọn đóng băng các lớp đầu để chỉ fine-tune các lớp cuối.

#     Args:
#         cfg (OmegaConf): Đối tượng cấu hình.
#         feature_extract (bool): Nếu True, đóng băng tất cả các lớp trừ lớp cuối.
#     """
#     model = None
#     num_ftrs = 0
    
#     print(f">>> Loading model: {cfg.MODEL.ARCH} -> useWeight: {useWeight}")
    
#     if cfg.MODEL.ARCH == 'resnet18':
#         weights = ResNet18_Weights.IMAGENET1K_V1 if useWeight else None
#         model = resnet18(weights=weights)
        
#         # Đóng băng các lớp nếu cần
#         # set_parameter_requires_grad(model, feature_extract)
        
#         # Thay thế lớp cuối và đảm bảo nó luôn có thể huấn luyện
#         num_ftrs = model.fc.in_features
#         model.fc = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(num_ftrs, numclasses),
#             # nn.Sigmoid()
#         )
#     elif cfg.MODEL.ARCH == 'resnet50':
#         weights = ResNet50_Weights.IMAGENET1K_V1 if useWeight else None
#         model = resnet50(weights=weights)
        
#         # Đóng băng các lớp nếu cần
#         # set_parameter_requires_grad(model, feature_extract)
        
#         # Thay thế lớp cuối và đảm bảo nó luôn có thể huấn luyện
#         num_ftrs = model.fc.in_features
#         model.fc = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(num_ftrs, numclasses),
#             # nn.Sigmoid()
#         )
#     elif cfg.MODEL.ARCH == 'mobilenet_v3_small':
#         weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if useWeight else None
#         model = mobilenet_v3_small(weights=weights)
        
#         set_parameter_requires_grad(model, feature_extract)
        
#         num_ftrs = model.classifier[-1].in_features
#         model.classifier[-1] = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(num_ftrs, numclasses),
#             # nn.Sigmoid()
#         )
#     elif cfg.MODEL.ARCH == 'densenet121':
#         weights = DenseNet121_Weights.IMAGENET1K_V1 if useWeight else None
#         model = densenet121(weights=weights)
        
#         # set_parameter_requires_grad(model, feature_extract)
        
#         # Lớp cuối của DenseNet có tên là 'classifier'
#         num_ftrs = model.classifier.in_features
#         model.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(num_ftrs, numclasses),
#             # nn.Sigmoid()
#         )
#     else:
#         raise ValueError(f"Model architecture {cfg.MODEL.ARCH} not supported.")

#     print(f"Model pre-trained on ImageNet loaded.")
#     if feature_extract:
#         print("Feature extracting mode: All layers frozen except the final classifier.")
#     else:
#         print("Fine-tuning mode: All layers are trainable.")
        
#     print(f"Model adapted for {numclasses} classes.")
    
#     return model

def get_model(cfg, feature_extract=False, useWeight=True, numclasses=5):
    """
    Tải mô hình và điều chỉnh lớp classifier cuối cùng một cách chính xác và nhất quán.
    """
    model = None
    arch = cfg.MODEL.ARCH.lower()
    
    print(f">>> Loading model: {arch} | useWeight: {useWeight} | num_classes: {numclasses}")
    
    # Bước 1: Tải mô hình gốc
    if arch == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1 if useWeight else None
        model = resnet18(weights=weights)
    elif arch == 'resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V1 if useWeight else None
        model = resnet50(weights=weights)
    elif arch == 'mobilenet_v3_small':
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if useWeight else None
        model = mobilenet_v3_small(weights=weights)
    elif arch == 'densenet121':
        weights = DenseNet121_Weights.IMAGENET1K_V1 if useWeight else None
        model = densenet121(weights=weights)
    else:
        raise ValueError(f"Model architecture {arch} not supported.")

    # Bước 2: Đóng băng (nếu cần)
    # if feature_extract:
    #     set_parameter_requires_grad(model, True)

    # Bước 3: Xác định số features và thay thế classifier
    if hasattr(model, 'fc'): # Dành cho ResNet
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, numclasses)
        )
    elif hasattr(model, 'classifier'): # Dành cho DenseNet và MobileNet
        # Xử lý DenseNet (classifier là một lớp Linear)
        if isinstance(model.classifier, nn.Linear):
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_ftrs, numclasses)
            )
        # Xử lý MobileNet (classifier là một Sequential)
        elif isinstance(model.classifier, nn.Sequential):
            num_ftrs = model.classifier[-1].in_features
            # Thay thế TOÀN BỘ classifier bằng một Sequential mới
            # Cách làm này đơn giản và hiệu quả hơn là chỉ thay lớp cuối
            model.classifier = nn.Sequential(
                nn.Linear(model.classifier[0].in_features, 512), # Lớp ẩn mới
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, numclasses)
            )
            # Hoặc cách đơn giản hơn chỉ thay lớp cuối
            # model.classifier[-1] = nn.Linear(num_ftrs, numclasses) # Sẽ giữ lại Dropout gốc
        else:
            raise TypeError(f"Unsupported classifier type: {type(model.classifier)}")
    else:
        raise AttributeError("Model does not have 'fc' or 'classifier' attribute.")


    print(f"Model pre-trained on ImageNet loaded.")
    if feature_extract:
        print("Feature extracting mode: All layers frozen except the final classifier.")
    else:
        print("Fine-tuning mode: All layers are trainable.")
        
    print(f"Model adapted for {numclasses} classes.")
    
    return model


def unfreeze_specific_layers(model, layers_to_unfreeze=['layer4', 'fc']):
    """
    "Mở băng" các lớp cụ thể để fine-tune.
    
    Args:
        model (nn.Module): Mô hình đã bị đóng băng.
        layers_to_unfreeze (list): Danh sách tên các lớp cần mở băng.
    """
    print(f"\nUnfreezing specific layers: {layers_to_unfreeze}")
    for name, param in model.named_parameters():
        for layer_name in layers_to_unfreeze:
            # Kiểm tra xem tên tham số có bắt đầu bằng tên lớp cần mở không
            # Ví dụ: 'layer4.0.conv1.weight' sẽ khớp với 'layer4'
            if name.startswith(layer_name):
                param.requires_grad = True
                break # Đã tìm thấy, không cần kiểm tra các tên lớp khác cho tham số này
    
    print("Trainable parameters after unfreezing:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

# --- Phần test file ---
if __name__ == '__main__':
    from omegaconf import OmegaConf
    
    # Tạo một config giả để test
    def create_test_config(arch_name):
        return OmegaConf.create({
            "MODEL": { "ARCH": arch_name, "NUM_CLASSES": 5 }
        })

    print("\n--- TESTING ResNet18 ---")
    cfg_resnet = create_test_config('resnet18')
    model_resnet = get_model(cfg_resnet)
    print(model_resnet.fc)

    print("\n" + "="*40)
    print("\n--- TESTING MobileNetV3-Small ---")
    cfg_mobile = create_test_config('mobilenet_v3_small')
    model_mobile = get_model(cfg_mobile)
    print(model_mobile.classifier)

    print("\n" + "="*40)
    print("\n--- TESTING DenseNet121 ---")
    cfg_dense = create_test_config('densenet121')
    model_dense = get_model(cfg_dense)
    print(model_dense.classifier)