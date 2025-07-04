# File: rotta_multilabel_adapter.py

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import logging
import random

from Medical.constants import COMMON_FINAL_LABEL_SET

# Import các thành phần cần thiết
# Giả sử chúng nằm trong cùng một package hoặc có đường dẫn đúng
from .base_adapter import BaseAdapter
from ..utils.memory_multilabel_advanced import AdvancedMultiLabelMemory

def timeliness_reweighting(ages, device):
    if not isinstance(ages, torch.Tensor):
        ages = torch.tensor(ages, dtype=torch.float32, device=device)
    return torch.exp(-ages / 100.0)

class RoTTAMultiLabel(BaseAdapter):
    def __init__(self, cfg, model, optimizer_func):
        # === BƯỚC 1: XÂY DỰNG MÔ HÌNH MỚI TRƯỚC ===
        # Vì BaseAdapter.__init__ sẽ gọi self.configure_model,
        # chúng ta cần self.model có cấu trúc đúng trước khi gọi super().__init__
        self.logger = logging.getLogger("TTA.adapter") # Khởi tạo logger trước
        self.cfg = cfg # Gán cfg trước
        
        # Tách backbone và tạo classifier mới có 6 lớp
        new_model = self.build_new_model_with_6_classes(model)
        
        # === BƯỚC 2: GỌI __init__ CỦA LỚP CHA VỚI MÔ HÌNH MỚI ===
        # Dòng này sẽ tự động:
        # 1. Gọi self.configure_model(new_model) -> gán self.model
        # 2. Gọi self.collect_params(self.model)
        # 3. Tạo self.optimizer
        # 4. Gán self.steps
        super().__init__(cfg, new_model, optimizer_func)
        
        # === BƯỚC 3: KHỞI TẠO CÁC THÀNH PHẦN CÒN LẠI CỦA ROTTA ===
        self.student = self.model # self.model đã được tạo bởi BaseAdapter
        self.teacher = self.build_ema(self.student)
        
        self.mem = AdvancedMultiLabelMemory(
            # capacity=cfg.ADAPTER.MEMORY_SIZE,
            cfg=cfg,
            num_classes=cfg.MODEL.NUM_CLASSES_TTA,
            class_names=COMMON_FINAL_LABEL_SET
            # lambda_t=cfg.ADAPTER.LAMBDA_T,
            # lambda_u=cfg.ADAPTER.LAMBDA_U
        )
        
        self.transform = nn.Identity()
        self.nu = 1.0 - cfg.ADAPTER.EMA_DECAY
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    # Trong file rotta_multilabel_adapter.py

    # Trong file rotta_multilabel_adapter.py

    def build_new_model_with_6_classes(self, source_model):
        """
        Tách backbone và tạo classifier mới, xử lý các cấu trúc classifier khác nhau,
        bao gồm cả trường hợp lồng nhau (nested).
        """
        self.logger.info("Rebuilding model: Separating backbone and creating a new classifier.")
        backbone = copy.deepcopy(source_model)
        num_ftrs = 0
        arch = self.cfg.MODEL.ARCH.lower()

        if arch.startswith('resnet'):
            if hasattr(backbone, 'fc') and isinstance(backbone.fc, nn.Sequential):
                # Giả định Linear là lớp cuối trong Sequential
                num_ftrs = backbone.fc[-1].in_features
                backbone.fc = nn.Identity()
            elif hasattr(backbone, 'fc') and isinstance(backbone.fc, nn.Linear):
                num_ftrs = backbone.fc.in_features
                backbone.fc = nn.Identity()
            else:
                raise TypeError(f"ResNet structure not as expected.")

        elif arch.startswith('densenet'):
            if hasattr(backbone, 'classifier') and isinstance(backbone.classifier, nn.Sequential):
                num_ftrs = backbone.classifier[-1].in_features
                backbone.classifier = nn.Identity()
            elif hasattr(backbone, 'classifier') and isinstance(backbone.classifier, nn.Linear):
                num_ftrs = backbone.classifier.in_features
                backbone.classifier = nn.Identity()
            else:
                raise TypeError(f"DenseNet structure not as expected.")

        elif arch.startswith('mobilenet'):
            if not hasattr(backbone, 'classifier') or not isinstance(backbone.classifier, nn.Sequential):
                raise TypeError(f"MobileNet structure not as expected.")
            
            # === PHẦN SỬA LỖI CHÍNH NẰM Ở ĐÂY ===
            # Lấy ra phần tử cuối cùng của classifier
            last_part = backbone.classifier[-1]

            # Kiểm tra xem phần tử cuối cùng có phải là một Sequential khác không
            if isinstance(last_part, nn.Sequential):
                self.logger.info("Detected nested Sequential in MobileNet classifier.")
                # Tìm lớp Linear bên trong Sequential lồng nhau này
                final_linear_layer = last_part[-1]
                if isinstance(final_linear_layer, nn.Linear):
                    num_ftrs = final_linear_layer.in_features
                    # Thay thế cả cái Sequential lồng nhau đó bằng Identity
                    backbone.classifier[-1] = nn.Identity()
                else:
                    raise TypeError("The last element of the nested Sequential is not a Linear layer.")
            
            # Nếu phần tử cuối cùng là một lớp Linear (trường hợp chuẩn)
            elif isinstance(last_part, nn.Linear):
                self.logger.info("Detected standard Linear layer in MobileNet classifier.")
                num_ftrs = last_part.in_features
                # Thay thế nó bằng Identity
                backbone.classifier[-1] = nn.Identity()
            else:
                raise TypeError(f"Unexpected layer type at the end of MobileNet classifier: {type(last_part)}")
                
        else:
            raise ValueError(f"Architecture {arch} is not supported.")

        if num_ftrs == 0:
            raise ValueError("Could not determine the number of input features for the new classifier.")

        # Tạo classifier mới có 6 lớp
        new_classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, self.cfg.MODEL.NUM_CLASSES_TTA)
        )
        
        # Kết hợp thành mô hình hoàn chỉnh mới
        new_model = nn.Sequential(backbone, new_classifier)
        
        self.logger.info(f"Successfully rebuilt model. New Classifier in_features: {num_ftrs}, out_features: {self.cfg.MODEL.NUM_CLASSES_TTA}")
        
        return new_model

    def configure_model(self, model: nn.Module):
        """
        Cấu hình các tham số có thể huấn luyện trên mô hình 6 lớp MỚI.
        """
        model.requires_grad_(False)
        
        # Logic mở băng giờ sẽ an toàn hơn
        # Chúng ta có thể chỉ mở băng classifier mới hoặc cả các lớp sâu hơn
        # model[0] là backbone, model[1] là new_classifier
        layers_to_unfreeze = ['0.layer4', '1'] # Mở băng layer4 của backbone và toàn bộ classifier mới
        
        self.logger.info(f"Unfreezing layers for TTA: {layers_to_unfreeze}")
        for name, param in model.named_parameters():
            for layer_name in layers_to_unfreeze:
                if name.startswith(layer_name):
                    param.requires_grad = True
                    self.logger.info(f"  - Unfroze {name}")
                    break
        return model

    @torch.no_grad()
    def get_teacher_output(self, x):
        """Lấy đầu ra từ teacher (mô hình 6 lớp)."""
        self.teacher.eval()
        teacher_logits = self.teacher(x)
        probs = torch.sigmoid(teacher_logits)
        pseudo_labels = (probs > 0.5).float()
        uncertainties = torch.mean(1 - torch.abs(probs - 0.5) * 2, dim=1)
        return pseudo_labels, uncertainties

    def forward_and_adapt(self, x, model, optimizer):
        """
        Triển khai logic cốt lõi TTA.
        """
        # === BƯỚC 1: SUY LUẬN VÀ THU THẬP THÔNG TIN ===
        pseudo_labels, mean_uncertainties = self.get_teacher_output(x)

        # === BƯỚC 2: CẬP NHẬT MEMORY BANK ===
        for i in range(x.size(0)):
            instance = (x[i], pseudo_labels[i], mean_uncertainties[i].item())
            self.mem.add_instance(instance)

        # === BƯỚC 3: HUẤN LUYỆN STUDENT ===
        bank_data, bank_labels, bank_ages = self.mem.get_memory()
        if len(bank_data) >= self.cfg.ADAPTER.BATCH_SIZE:
            model.train()
            
            indices = random.sample(range(len(bank_data)), self.cfg.ADAPTER.BATCH_SIZE)
            
            batch_images = torch.stack([bank_data[i] for i in indices]).to(x.device)
            batch_labels = torch.stack([bank_labels[i] for i in indices]).to(x.device)
            batch_ages = [bank_ages[i] for i in indices]

            strong_aug_images = self.transform(batch_images)
            student_logits = model(strong_aug_images)
            
            instance_loss = self.criterion(student_logits, batch_labels).mean(dim=1)
            instance_weight = timeliness_reweighting(batch_ages, device=x.device)
            final_loss = (instance_loss * instance_weight).mean()
            
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            
            # === BƯỚC 4: CẬP NHẬT TEACHER ===
            self.update_ema_variables()

        # Trả về đầu ra của Teacher để đánh giá
        with torch.no_grad():
            self.teacher.eval()
            final_output = self.teacher(x)
        return final_output
    
    def update_ema_variables(self):
        """
        Cập nhật Teacher (self.teacher) từ Student (self.student) bằng EMA.
        Phương thức này thuộc về đối tượng RoTTAMultiLabel.
        """
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            # nu là tỷ lệ cho trọng số MỚI (từ student)
            teacher_param.data.mul_(1.0 - self.nu).add_(student_param.data, alpha=self.nu)