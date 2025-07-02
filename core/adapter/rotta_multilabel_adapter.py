import torch
import torch.nn as nn
import torch.optim as optim
import copy
import logging
import random

# Import các thành phần cần thiết
# from .base_adapter import BaseAdapter # Nếu bạn có file này
from ..utils.memory_multilabel_cstu import CSTUMultiLabel
# from ..utils.custom_transforms import get_tta_transforms
# from ..utils.utils import ...

# Hàm helper cho reweighting
def timeliness_reweighting(ages, device):
    if not isinstance(ages, torch.Tensor):
        ages = torch.tensor(ages).float().to(device)
    # Hàm sigmoid ngược để trọng số giảm khi tuổi tăng
    return torch.exp(-ages / 100.0) # Chia cho một hằng số để làm mượt

# Kế thừa từ nn.Module để đơn giản hóa, bạn có thể đổi thành BaseAdapter nếu cần
class RoTTAMultiLabel(nn.Module): 
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.logger = logging.getLogger("TTA.adapter") # Sử dụng logger nếu có
        
        # Cấu hình model, chỉ mở băng các lớp cần thiết
        self.model = self.configure_model(copy.deepcopy(model))
        
        # Khởi tạo Student và Teacher
        self.student = self.model
        self.teacher = self.build_ema(self.model)
        
        # Thu thập các tham số có thể huấn luyện và tạo optimizer
        params_to_train, _ = self.collect_params(self.student)
        self.optimizer = optim.Adam(params_to_train, lr=cfg.ADAPTER.LR, weight_decay=cfg.OPTIM.WD)
        
        # Khởi tạo memory bank CSTU cho đa nhãn
        self.mem = CSTUMultiLabel(
            capacity=cfg.ADAPTER.MEMORY_SIZE,
            num_class=cfg.MODEL.NUM_CLASSES,
            lambda_t=1.0,
            lambda_u=1.0
        )
        
        # Các thuộc tính khác
        # self.transform = get_tta_transforms(cfg)
        self.transform = nn.Identity() # Tạm dùng transform đơn giản
        self.nu = 1.0 - cfg.ADAPTER.EMA_DECAY # nu trong paper là tỷ lệ cho trọng số MỚI
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def configure_model(self, model: nn.Module):
        """Cấu hình các tham số có thể huấn luyện."""
        model.requires_grad_(False)
        
        layers_to_unfreeze = []
        arch = self.cfg.MODEL.ARCH.lower()
        if arch.startswith('resnet'):
            layers_to_unfreeze = ['layer4', 'fc']
        elif arch.startswith('mobilenet'):
            layers_to_unfreeze = ['features.12', 'classifier']
        elif arch.startswith('densenet'):
            layers_to_unfreeze = ['features.denseblock4', 'features.norm5', 'classifier']
        
        self.logger.info(f"Unfreezing layers: {layers_to_unfreeze}")
        for name, param in model.named_parameters():
            for layer_name in layers_to_unfreeze:
                if name.startswith(layer_name):
                    param.requires_grad = True
                    break
                    
        return model

    def collect_params(self, model: nn.Module):
        """Thu thập các tham số có requires_grad=True."""
        params = []
        names = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params.append(param)
                names.append(name)
        return params, names

    @staticmethod
    def build_ema(model):
        """Tạo một bản sao của mô hình để làm EMA/Teacher."""
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        return ema_model

    @torch.no_grad()
    def get_teacher_output(self, x):
        """Lấy đầu ra từ teacher và các thông tin cần thiết."""
        self.teacher.eval()
        teacher_logits = self.teacher(x)
        probs = torch.sigmoid(teacher_logits)
        
        pseudo_labels = (probs > 0.5).float()
        
        # Uncertainty: Càng gần 0.5, uncertainty càng cao.
        uncertainties = 1 - torch.abs(probs - 0.5) * 2
        # Lấy uncertainty trung bình trên các lớp
        mean_uncertainties = torch.mean(uncertainties, dim=1)
        
        return pseudo_labels, mean_uncertainties

    def forward(self, x):
        """
        Triển khai logic `forward_and_adapt` trong một hàm duy nhất.
        """
        # === BƯỚC 1: SUY LUẬN VÀ THU THẬP THÔNG TIN ===
        pseudo_labels, mean_uncertainties = self.get_teacher_output(x)

        # === BƯỚC 2: CẬP NHẬT MEMORY BANK ===
        for i in range(x.size(0)):
            instance = (x[i], pseudo_labels[i], mean_uncertainties[i].item())
            self.mem.add_instance(instance)

        # === BƯỚC 3: HUẤN LUYỆN STUDENT ===
        # Lấy dữ liệu từ memory bank
        bank_data, bank_labels, bank_ages = self.mem.get_memory()

        if len(bank_data) >= self.cfg.ADAPTER.BATCH_SIZE:
            self.student.train()
            self.teacher.train()

            # Lấy một batch ngẫu nhiên từ toàn bộ memory
            indices = random.sample(range(len(bank_data)), self.cfg.ADAPTER.BATCH_SIZE)
            
            # Lấy dữ liệu cho batch
            batch_images = torch.stack([bank_data[i] for i in indices]).to(x.device)
            batch_labels = torch.stack([bank_labels[i] for i in indices]).to(x.device)
            batch_ages = [bank_ages[i] for i in indices]

            # Tăng cường dữ liệu mạnh (nếu có)
            strong_aug_images = self.transform(batch_images)

            # Cập nhật student
            student_logits = self.student(strong_aug_images)
            
            # Tính loss
            instance_loss = self.criterion(student_logits, batch_labels).mean(dim=1)
            instance_weight = timeliness_reweighting(batch_ages, device=x.device)
            final_loss = (instance_loss * instance_weight).mean()
            
            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()
            
            # === BƯỚC 4: CẬP NHẬT TEACHER ===
            self.update_ema_variables()

        # Trả về đầu ra của Teacher (ổn định hơn) để đánh giá
        with torch.no_grad():
            self.teacher.eval()
            final_output = self.teacher(x)

        return final_output

    def update_ema_variables(self):
        """Cập nhật Teacher từ Student bằng EMA."""
        for ema_param, param in zip(self.teacher.parameters(), self.student.parameters()):
            ema_param.data.mul_(1.0 - self.nu).add_(param.data, alpha=self.nu)