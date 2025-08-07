import torch
import torch.nn as nn
from copy import deepcopy


class MomentumBN(nn.Module):
    def __init__(self, bn_layer: nn.BatchNorm2d, momentum):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum
        if bn_layer.track_running_stats and bn_layer.running_var is not None and bn_layer.running_mean is not None:
            self.register_buffer("source_mean", deepcopy(bn_layer.running_mean))
            self.register_buffer("source_var", deepcopy(bn_layer.running_var))
            self.source_num = bn_layer.num_batches_tracked
        self.weight = deepcopy(bn_layer.weight)
        self.bias = deepcopy(bn_layer.bias)

        self.register_buffer("target_mean", torch.zeros_like(self.source_mean))
        self.register_buffer("target_var", torch.ones_like(self.source_var))
        self.eps = bn_layer.eps

        self.current_mu = None
        self.current_sigma = None

    def forward(self, x):
        raise NotImplementedError


class RobustBN1d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=0, unbiased=False, keepdim=False)  # (C,)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(var.detach())
            mean, var = mean.view(1, -1), var.view(1, -1)
        else:
            mean, var = self.source_mean.view(1, -1), self.source_var.view(1, -1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1)
        bias = self.bias.view(1, -1)

        return x * weight + bias


class RobustBN2d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=False)  # (C,)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(var.detach())
            mean, var = mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1)
        else:
            mean, var = self.source_mean.view(1, -1, 1, 1), self.source_var.view(1, -1, 1, 1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        return x * weight + bias



import torch.nn.functional as F
class RobustBN(nn.Module):
    """
    A Robust BatchNorm layer that maintains source statistics and updates them
    with an exponential moving average during adaptation (training mode).
    In evaluation mode, it uses the current statistics.
    """
    def __init__(self, bn_layer, momentum):
        super().__init__()
        # Sao chép các thuộc tính cần thiết từ lớp BN gốc
        self.num_features = bn_layer.num_features
        self.momentum = momentum
        self.eps = bn_layer.eps
        
        # Sao chép các tham số có thể học
        self.weight = nn.Parameter(deepcopy(bn_layer.weight.data))
        self.bias = nn.Parameter(deepcopy(bn_layer.bias.data))
        
        # Đăng ký các buffer để lưu trữ thống kê
        # Đây là các thống kê "hiện tại" của TTA, được cập nhật liên tục
        self.register_buffer("running_mean", deepcopy(bn_layer.running_mean))
        self.register_buffer("running_var", deepcopy(bn_layer.running_var))
        
        # Luôn bật chế độ theo dõi để buffer được lưu trong state_dict
        self.track_running_stats = True
        
    def forward(self, x):
        # Khi model ở chế độ train (tức là trong lúc update_model của RoTTA)
        if self.training:
            # 1. Tính toán thống kê của batch hiện tại
            # Dùng ước lượng không chệch (unbiased) để nhất quán với nn.BatchNorm
            batch_mean = x.mean(dim=[0, 2, 3])
            batch_var = x.var(dim=[0, 2, 3], unbiased=True)
            
            # 2. Cập nhật running_mean và running_var bằng EMA
            # Cập nhật tại chỗ (in-place) để hiệu quả
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * batch_mean
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * batch_var
        
        # Dù ở chế độ train hay eval, chúng ta luôn dùng running_mean và running_var đã cập nhật
        # Điều này khác với BN chuẩn, nhưng đúng với ý tưởng của RoTTA (luôn dùng thống kê mới nhất)
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=False, # Luôn dùng running stats để chuẩn hóa, không dùng batch stats
            momentum=0,     # Không để F.batch_norm tự cập nhật running stats
            eps=self.eps
        )