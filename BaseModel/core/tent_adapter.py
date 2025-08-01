# File: core/adapter/tent_adapter.py

import torch
import torch.nn as nn
import torch.optim as optim
import logging

from .base_adapter import BaseAdapter

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy của đầu ra softmax."""
    # Áp dụng cho bài toán đa nhãn, ta tính entropy trên từng đầu ra sigmoid
    # và lấy trung bình.
    # p * log(p) + (1-p) * log(1-p)
    p = torch.sigmoid(x)
    return - (p * torch.log(p + 1e-6) + (1-p) * torch.log(1-p + 1e-6)).mean(dim=1)


class TENT(BaseAdapter):
    def __init__(self, cfg, model, optimizer_func):
        super(BaseAdapter, self).__init__()
        self.logger = logging.getLogger("TTA.adapter")
        self.cfg = cfg
        
        # TENT chỉ cập nhật các lớp BatchNorm
        self.model = self.configure_model(model)
        
        # Gọi __init__ của BaseAdapter với mô hình đã được cấu hình
        # Nó sẽ tự tạo optimizer chỉ với các tham số của BatchNorm
        super().__init__(cfg, self.model, optimizer_func)
        
        self.logger.info("TENT adapter initialized.")

    def configure_model(self, model: nn.Module):
        """
        Cấu hình mô hình cho TENT: chỉ mở băng các tham số của BatchNorm.
        """
        model.requires_grad_(False)
        
        self.logger.info("Configuring model for TENT: Making BatchNorm layers trainable.")
        trainable_param_names = []
        for name, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                m.requires_grad_(True)
                # TENT cập nhật cả thống kê, nên đặt ở chế độ train
                m.train() 
                for param_name, _ in m.named_parameters():
                    trainable_param_names.append(f"{name}.{param_name}")

        self.logger.info(f"Trainable parameters set: {trainable_param_names}")
        return model

    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        """
        Thực hiện forward pass và cập nhật bằng cách tối thiểu hóa entropy.
        """
        if optimizer is None:
            return model(x) # Nếu không có gì để train, chỉ forward

        model.train() # Đặt model ở chế độ train để BN cập nhật thống kê
        
        outputs = model(x) # Lấy logits
        
        # Tính loss là entropy
        loss = softmax_entropy(outputs).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Trả về output để đánh giá
        return outputs