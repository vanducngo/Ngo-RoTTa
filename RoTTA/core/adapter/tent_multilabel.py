# core/adapter/tent_multilabel.py

import torch
import torch.nn as nn
from copy import deepcopy
from .base_adapter import BaseAdapter

class TentMultiLabel(BaseAdapter):
    """
    TENT for Multi-Label Classification, adapted from the original implementation.
    Adapts a model by minimizing entropy of the sigmoid outputs.
    """
    def __init__(self, cfg, model, optimizer_func):
        super().__init__(cfg, model, optimizer_func)
        self.steps = self.cfg.OPTIM.STEPS
        assert self.steps > 0, "TENT requires >= 1 step(s) to forward and update"

    def configure_model(self, model: nn.Module):
        """
        Configure model for use with TENT, following the original implementation.
        - Set model to train mode.
        - Disable gradients for all parameters.
        - Enable gradients and disable running stats for normalization layers.
        """
        # train mode is essential for TENT, as it uses batch statistics
        model.train()
        # disable gradients for all parameters
        model.requires_grad_(False)
        
        # enable gradients for affine parameters in normalization layers
        # and disable usage of running stats
        for m in model.modules():
            # TENT gốc chỉ xử lý BatchNorm2d, ta mở rộng cho các loại Norm khác
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                # Đây là một chi tiết RẤT QUAN TRỌNG của TENT
                if hasattr(m, 'track_running_stats'):
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
        return model

    @staticmethod
    def collect_params(model):
        """
        Collect the affine scale + shift parameters from normalization layers.
        Bám sát `collect_params` của TENT gốc, nhưng tổng quát hơn.
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
                # Lấy tất cả các tham số có thể học của lớp Norm (thường là weight và bias)
                for np, p in m.named_parameters():
                    if p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        """
        Forward pass and adaptation step for multi-label TENT.
        """
        # Forward pass
        outputs = model(x) # Logits

        # Adaptation step
        # Tối thiểu hóa entropy của đầu ra sigmoid
        loss = sigmoid_entropy_u_shaped(outputs).mean()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return outputs

# --- HÀM ENTROPY ĐÃ ĐƯỢC CẢI TIẾN ---
@torch.jit.script
def sigmoid_entropy_u_shaped(logits: torch.Tensor) -> torch.Tensor:
    """
    Entropy function for sigmoid outputs, designed to be minimized.
    This function is U-shaped, with a minimum at p=0.5.
    Entropy = - (p * log(p) + (1-p) * log(1-p)) is not ideal as it's not symmetric around 0.5.
    
    A better alternative is to measure the distance from the decision boundary (0.5).
    This encourages probabilities to move towards 0 or 1.
    
    This is equivalent to minimizing: -(2p - 1)^2  or maximizing (2p - 1)^2
    Let's use a variant of binary entropy that is symmetric:
    H(p) = 1 - (2p - 1)^2 = 4p(1-p)
    This is minimized when p is 0 or 1.
    """
    probs = torch.sigmoid(logits) # Không cần detach() ở đây
    
    # Sử dụng hàm loss H(p) = 4 * p * (1-p)
    # Hàm này có giá trị min = 0 tại p=0 và p=1, max = 1 tại p=0.5
    # Tối thiểu hóa hàm này sẽ đẩy xác suất về 0 hoặc 1.
    entropy = 4 * probs * (1 - probs)
    
    # Tổng entropy trên các lớp
    return torch.sum(entropy, dim=1)