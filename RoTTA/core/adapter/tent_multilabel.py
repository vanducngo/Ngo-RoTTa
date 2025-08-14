import torch
import torch.nn as nn
from .base_adapter import BaseAdapter

class TentMultiLabel(BaseAdapter):
    def __init__(self, cfg, model, optimizer_func):
        super().__init__(cfg, model, optimizer_func)
        self.steps = self.cfg.OPTIM.STEPS
        assert self.steps > 0, "TENT requires >= 1 step(s) to forward and update"

    def configure_model(self, model: nn.Module):
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
        # Forward pass
        outputs = model(x) # Logits

        # Adaptation step
        # Tối thiểu hóa entropy của đầu ra sigmoid
        loss = sigmoid_entropy_u_shaped(outputs).mean()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return outputs

@torch.jit.script
def sigmoid_entropy_u_shaped(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    entropy = 4 * probs * (1 - probs)
    # Tổng entropy trên các lớp
    return torch.sum(entropy, dim=1)