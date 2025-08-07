import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from .base_adapter import BaseAdapter
from ..utils.custom_transforms import get_tta_transforms 

class CoTTAMultiLabel(BaseAdapter):
    """
    CoTTA for Multi-Label Classification.
    This implementation adapts the original CoTTA for a multi-label setting
    and integrates it into the project's BaseAdapter framework.
    """
    def __init__(self, cfg, model, optimizer_func):
        # BaseAdapter __init__ sẽ gọi self.configure_model và self.collect_params
        super().__init__(cfg, model, optimizer_func)
        self.steps = self.cfg.OPTIM.STEPS
        
        # Lưu các tham số đặc trưng của CoTTA
        self.mt_alpha = self.cfg.OPTIM.MT
        self.rst_prob = self.cfg.OPTIM.RST
        self.ap_threshold = self.cfg.OPTIM.AP
        
        # Khởi tạo các mô hình cần thiết
        # self.model là student model (từ BaseAdapter), đã ở đúng device
        self.model_state, self.optimizer_state, self.teacher, self.anchor = \
            self._copy_model_and_optimizer(self.model, self.optimizer)

        # CoTTA sử dụng augmentation mạnh để tạo nhãn giả
        # get_tta_transforms cần được định nghĩa ở đâu đó trong project của bạn
        self.transform = get_tta_transforms(self.cfg) 
        self.num_augmentations = 32 # Giữ nguyên như bản gốc

    def configure_model(self, model: nn.Module):
        """Configure model for CoTTA: enable training for all parameters."""
        model.train()
        model.requires_grad_(True) 
        
        # Cấu hình riêng cho BN để dùng batch stats, giống TENT/CoTTA gốc
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if hasattr(m, 'track_running_stats'):
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
        self.logger.info("Configured model for CoTTA: All parameters are trainable, BN layers use batch stats.")
        return model

    @staticmethod
    def collect_params(model):
        """CoTTA updates all trainable parameters."""
        params = [p for p in model.parameters() if p.requires_grad]
        return params, [name for name, p in model.named_parameters() if p.requires_grad]
    
    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        # 1. Tạo nhãn giả "mềm" chất lượng cao từ Teacher
        with torch.no_grad():
            # Sử dụng anchor model để quyết định có dùng augmentation averaging không
            anchor_probs = torch.sigmoid(self.anchor(x))
            # Confidence cho đa nhãn: trung bình khoảng cách đến điểm không chắc chắn 0.5
            confidence = torch.mean(torch.abs(anchor_probs - 0.5) * 2, dim=1)
            
            # Dự đoán trên ảnh gốc từ teacher
            teacher_logits = self.teacher(x)

            # Nếu độ tự tin trung bình của batch thấp, dùng augmentation averaging
            if confidence.mean() < self.ap_threshold:
                # Tạo N phiên bản augment và lấy dự đoán trung bình
                aug_logits_list = [self.teacher(self.transform(x)) for _ in range(self.num_augmentations)]
                avg_logits = torch.stack(aug_logits_list).mean(dim=0)
                # Kết hợp dự đoán gốc và dự đoán augment để ổn định hơn
                final_teacher_logits = (teacher_logits + avg_logits) / 2
            else:
                final_teacher_logits = teacher_logits
        
        # 2. Cập nhật Student model
        student_logits = model(x)
        loss = F.binary_cross_entropy_with_logits(student_logits, torch.sigmoid(final_teacher_logits.detach()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # 3. Cập nhật Teacher model bằng EMA
        self._update_ema_variables(self.teacher, model, self.mt_alpha)

        # 4. Stochastic Restoration (PHIÊN BẢN ĐÃ SỬA LỖI)
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Lấy tham số gốc tương ứng từ state_dict bằng tên (key)
                if name in self.model_state:
                    state_param = self.model_state[name]
                    
                    # Tạo mask ngẫu nhiên
                    mask = (torch.rand_like(param) < self.rst_prob).float()
                    
                    with torch.no_grad():
                        # Thực hiện restore
                        param.data = state_param.to(param.device) * mask + param.data * (1. - mask)
                else:
                    self.logger.warning(f"Parameter '{name}' not found in the initial model state. Skipping stochastic restoration for it.")
        
        # Trả về dự đoán của teacher (ổn định hơn) để đánh giá
        return final_teacher_logits

    def _copy_model_and_optimizer(self, model, optimizer):
        """Tạo các bản sao cần thiết cho CoTTA."""
        model_state = deepcopy(model.state_dict())
        optimizer_state = deepcopy(optimizer.state_dict()) if optimizer is not None else None
        
        teacher = deepcopy(model)
        teacher.requires_grad_(False); teacher.eval()
        
        anchor = deepcopy(model)
        anchor.requires_grad_(False); anchor.eval()
        
        return model_state, optimizer_state, teacher, anchor

    @staticmethod
    def _update_ema_variables(ema_model, model, alpha):
        """Cập nhật trọng số của Teacher model."""
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            # Công thức EMA: ema = alpha * ema + (1 - alpha) * current
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)