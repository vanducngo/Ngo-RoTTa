import torch.nn as nn
from .base_adapter import BaseAdapter

class NormMultiLabel(BaseAdapter):
    def __init__(self, cfg, model, optimizer_func):
        super().__init__(cfg, model, optimizer_func)
        self.logger.info("NormMultiLabel adapter initialized.")

    def configure_model(self, model: nn.Module):
        self.logger.info("Configuring model for NORM: Setting normalization layers to train mode.")
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
                m.train()
        return model

    def collect_params(self, model: nn.Module):
        return [], []

    def forward_and_adapt(self, x, model, optimizer):
        return model(x)