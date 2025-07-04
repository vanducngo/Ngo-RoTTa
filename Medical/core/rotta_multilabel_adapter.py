import torch
import torch.nn as nn
import torch.optim as optim
import copy
import logging
import random

from constants import COMMON_FINAL_LABEL_SET, TARGET_INDICES_IN_FULL_LIST

from .base_adapter import BaseAdapter
from .cstu_multilabel import CSTUMultiLabel 
# (Đảm bảo đường dẫn import này đúng với cấu trúc thư mục của bạn)

def timeliness_reweighting(ages, device):
    if not isinstance(ages, torch.Tensor):
        ages = torch.tensor(ages, dtype=torch.float32, device=device)
    return torch.exp(-ages / 100.0)

class RoTTAMultiLabel(BaseAdapter):
    def __init__(self, cfg, model, optimizer_func):
        # Khởi tạo logger và cfg trước
        self.logger = logging.getLogger("TTA.adapter")
        self.cfg = cfg

        # Xây dựng mô hình mới (backbone + classifier 6 lớp)
        # Hàm này phải trả về một mô hình hợp lệ
        new_model = self.build_new_model_with_6_classes(model)
        
        # Gọi __init__ của BaseAdapter với mô hình mới
        super().__init__(cfg, new_model, optimizer_func)

        # Khởi tạo các thành phần của RoTTA
        self.student = self.model
        self.teacher = self.build_ema(self.student)
        
        self.mem = CSTUMultiLabel(
            capacity=self.cfg.ADAPTER.MEMORY_SIZE,
            num_class=self.cfg.MODEL.NUM_CLASSES_TTA,
            lambda_t=self.cfg.ADAPTER.LAMBDA_T,
            lambda_u=self.cfg.ADAPTER.LAMBDA_U
        )
        
        self.transform = nn.Identity()
        self.nu = 1.0 - self.cfg.ADAPTER.EMA_DECAY
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.update_frequency = self.cfg.ADAPTER.UPDATE_FREQUENCY
        self.instance_counter = 0

    def build_new_model_with_6_classes(self, source_model):
        """
        Tách backbone từ mô hình nguồn và gắn một classifier 6 lớp mới.
        Đây là phiên bản đã được sửa lỗi.
        """
        self.logger.info("Rebuilding source model for TTA...")
        backbone = copy.deepcopy(source_model)
        num_ftrs = 0
        arch = self.cfg.MODEL.ARCH.lower()

        if arch.startswith('resnet') or arch.startswith('densenet'):
            classifier_name = 'fc' if hasattr(backbone, 'fc') else 'classifier'
            if hasattr(backbone, classifier_name):
                classifier = getattr(backbone, classifier_name)
                # Xử lý cả trường hợp classifier là Linear hoặc Sequential
                if isinstance(classifier, nn.Sequential):
                    # Giả định lớp Linear nằm ở cuối
                    num_ftrs = classifier[-1].in_features
                elif isinstance(classifier, nn.Linear):
                    num_ftrs = classifier.in_features
                else:
                     raise TypeError(f"Unsupported classifier type in {arch}: {type(classifier)}")
                # Thay thế bằng Identity để tách backbone
                setattr(backbone, classifier_name, nn.Identity())
            else:
                raise AttributeError(f"Model {arch} does not have a 'fc' or 'classifier' attribute.")
        
        elif arch.startswith('mobilenet'):
            if hasattr(backbone, 'classifier') and isinstance(backbone.classifier, nn.Sequential):
                last_part_of_classifier = backbone.classifier[-1]
                if isinstance(last_part_of_classifier, nn.Linear):
                    num_ftrs = last_part_of_classifier.in_features
                    backbone.classifier[-1] = nn.Identity()
                elif isinstance(last_part_of_classifier, nn.Sequential): # Trường hợp lồng nhau
                    num_ftrs = last_part_of_classifier[-1].in_features
                    backbone.classifier[-1] = nn.Identity()
                else:
                    raise TypeError("Last part of MobileNet classifier is not Linear or Sequential.")
            else:
                 raise TypeError("MobileNet classifier is not a Sequential module as expected.")
        
        else:
            raise ValueError(f"Architecture {arch} is not supported for automatic backbone separation.")

        if num_ftrs == 0:
            raise ValueError("Could not determine the number of input features for the new classifier.")

        # Tạo classifier mới
        new_classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, self.cfg.MODEL.NUM_CLASSES_TTA)
        )
        
        # Kết hợp thành mô hình hoàn chỉnh mới
        new_model = nn.Sequential(backbone, new_classifier)
        self.logger.info(f"Successfully rebuilt model. New classifier has {num_ftrs} in_features.")
        
        return new_model # Đảm bảo luôn trả về một mô hình

    def configure_model(self, model: nn.Module):
        """Cấu hình các tham số có thể huấn luyện."""
        self.logger.info("Configuring model: Making BatchNorm layers trainable.")
        model.requires_grad_(False) # Dòng này sẽ không còn lỗi
        
        for name, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                m.requires_grad_(True)
        
        # Mở băng cả classifier mới
        if hasattr(model, '1') and isinstance(model[1], nn.Sequential):
            for param in model[1].parameters():
                param.requires_grad = True
                
        return model

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        # 1. Teacher dự đoán
        with torch.no_grad():
            model.eval()
            self.teacher.eval()
            
            teacher_logits = self.teacher(batch_data)
            probs = torch.sigmoid(teacher_logits)
            pseudo_labels = (probs > 0.5).float()
            
            # Tính uncertainty cho mỗi mẫu
            uncertainties = torch.mean(1 - torch.abs(probs - 0.5) * 2, dim=1)

        # 2. Thêm vào memory
        for i in range(batch_data.size(0)):
            instance = (batch_data[i], pseudo_labels[i], uncertainties[i].item())
            self.mem.add_instance(instance)
            self.instance_counter += 1

            # 3. Cập nhật mô hình theo tần suất
            if self.instance_counter % self.update_frequency == 0:
                self.update_model(model, optimizer)
        
        return teacher_logits

    def update_model(self, model, optimizer):
        device = torch.device(self.cfg.TRAINING.DEVICE if torch.cuda.is_available() else "cpu")
        model.train()
        self.teacher.train()
        
        bank_data, bank_labels, bank_ages = self.mem.get_memory()
        
        if len(bank_data) < self.cfg.ADAPTER.BATCH_SIZE:
            return

        # Lấy một batch ngẫu nhiên
        indices = random.sample(range(len(bank_data)), self.cfg.ADAPTER.BATCH_SIZE)
        batch_images = torch.stack([bank_data[i] for i in indices]).to(device)
        batch_labels = torch.stack([bank_labels[i] for i in indices]).to(device)
        batch_ages = [bank_ages[i] for i in indices]

        strong_aug_images = self.transform(batch_images)
        
        # Dùng Teacher để tạo target "mềm" (soft targets)
        with torch.no_grad():
            teacher_sup_logits = self.teacher(batch_images)
        
        student_sup_logits = model(strong_aug_images)
        
        # Loss: So sánh đầu ra của student với nhãn giả "cứng"
        instance_loss = self.criterion(student_sup_logits, batch_labels).mean(dim=1)
        
        # Áp dụng trọng số
        instance_weight = timeliness_reweighting(batch_ages, device=device)
        final_loss = (instance_loss * instance_weight).mean()
        
        if final_loss is not None:
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

        self.update_ema_variables(self.teacher, model, self.nu)

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(1.0 - nu).add_(param.data, alpha=nu)

class RoTTAMultiLabelSelective(BaseAdapter): # Đổi tên để phân biệt
    def __init__(self, cfg, model, optimizer_func):
        self.logger = logging.getLogger("TTA.adapter")
        self.cfg = cfg
        
        # SỬ DỤNG TRỰC TIẾP MÔ HÌNH 14 LỚP
        # Không cần build_new_model nữa
        super().__init__(cfg, model, optimizer_func)

        self.student = self.model
        self.teacher = self.build_ema(self.student)
        
        # Memory bank vẫn hoạt động trên không gian 6 lớp
        self.mem = CSTUMultiLabel(
            capacity=cfg.ADAPTER.MEMORY_SIZE,
            num_class=len(COMMON_FINAL_LABEL_SET), # Số lớp là 6
            lambda_t=cfg.ADAPTER.LAMBDA_T,
            lambda_u=cfg.ADAPTER.LAMBDA_U
        )
        
        self.transform = nn.Identity()
        self.nu = 1.0 - cfg.ADAPTER.EMA_DECAY
        # Loss vẫn là BCE, nhưng sẽ được áp dụng có chọn lọc
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        # Lưu lại các chỉ số của 6 lớp mục tiêu
        device = torch.device(self.cfg.TRAINING.DEVICE if torch.cuda.is_available() else "cpu")
        self.target_indices = torch.tensor(TARGET_INDICES_IN_FULL_LIST, device=device)

    def configure_model(self, model: nn.Module):
        """Cấu hình các tham số có thể huấn luyện trên mô hình 14 lớp."""
        model.requires_grad_(False)
        
        # Mở băng các lớp BatchNorm (chiến lược an toàn của RoTTA gốc)
        self.logger.info("Configuring model: Making BatchNorm layers and final classifier trainable.")
        for name, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                m.requires_grad = True

        # Mở băng cả lớp classifier 14 lớp cuối cùng
        # Vì chúng ta cần cập nhật trọng số cho cả 14 đầu ra,
        # ngay cả khi loss chỉ tính trên 6
        if hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
             for param in model.classifier.parameters():
                param.requires_grad = True
        return model

    @torch.no_grad()
    def get_teacher_output_selective(self, x):
        """
        Lấy đầu ra từ teacher, sau đó lọc ra 6 lớp để tạo nhãn giả và uncertainty.
        """
        self.teacher.eval()
        teacher_logits_14_cls = self.teacher(x) # Logits này có kích thước [batch, 14]
        
        # === LỌC RA 6 LỚP ===
        teacher_logits_6_cls = torch.index_select(teacher_logits_14_cls, 1, self.target_indices)

        # Tính toán trên đầu ra 6 lớp
        probs = torch.sigmoid(teacher_logits_6_cls)
        pseudo_labels = (probs > 0.5).float() # Kích thước [batch, 6]
        
        uncertainties = torch.mean(1 - torch.abs(probs - 0.5) * 2, dim=1)
        return pseudo_labels, uncertainties

    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        # 1. Tạo nhãn giả 6 lớp
        pseudo_labels, mean_uncertainties = self.get_teacher_output_selective(x)

        # 2. Cập nhật Memory Bank
        for i in range(x.size(0)):
            instance = (x[i], pseudo_labels[i], mean_uncertainties[i].item())
            self.mem.add_instance(instance)

        # 3. Huấn luyện Student
        # Lấy TOÀN BỘ dữ liệu từ memory bank
        bank_data, bank_labels, bank_ages = self.mem.get_memory()
        
        if len(bank_data) >= self.cfg.ADAPTER.BATCH_SIZE:
            model.train()
            
            # Lấy một BATCH ngẫu nhiên từ toàn bộ memory
            indices = random.sample(range(len(bank_data)), self.cfg.ADAPTER.BATCH_SIZE)
            
            # Tạo batch dữ liệu từ các chỉ số đã chọn
            batch_images = torch.stack([bank_data[i] for i in indices]).to(x.device)
            batch_labels_6_cls = torch.stack([bank_labels[i] for i in indices]).to(x.device)
            
            # === PHẦN SỬA LỖI ===
            # Lấy tuổi tương ứng với các mẫu trong batch
            batch_ages_list = [bank_ages[i] for i in indices]
            # ===================

            student_logits_14_cls = model(batch_images)
            
            # Lọc ra 6 logits tương ứng từ đầu ra của student
            student_logits_6_cls = torch.index_select(student_logits_14_cls, 1, self.target_indices)
            
            # Tính loss chỉ trên 6 logits này với 6 nhãn giả
            instance_loss = self.criterion(student_logits_6_cls, batch_labels_6_cls).mean(dim=1)
            
            # === PHẦN SỬA LỖI ===
            # Tính trọng số dựa trên tuổi của BATCH, không phải toàn bộ memory
            instance_weight = timeliness_reweighting(batch_ages_list, device=x.device)
            # ===================
            
            # Giờ đây, instance_loss và instance_weight đều sẽ có kích thước [32]
            final_loss = (instance_loss * instance_weight).mean()
            
            if optimizer is not None:
                optimizer.zero_grad()
                final_loss.backward()
                optimizer.step()
            
            self.update_ema_variables(self.teacher, self.student, self.nu)

        # 4. Trả về kết quả để đánh giá
        with torch.no_grad():
            self.teacher.eval()
            final_output_14_cls = self.teacher(x)
            final_output_6_cls = torch.index_select(final_output_14_cls, 1, self.target_indices)
            
        return final_output_6_cls
    
    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(1.0 - nu).add_(param.data, alpha=nu)