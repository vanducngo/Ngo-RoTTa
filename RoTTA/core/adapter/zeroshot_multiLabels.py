# RoTTA.py

import torch
import torch.nn as nn
from .base_adapter import BaseAdapter

class ZeroshotMultiLabels(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super(ZeroshotMultiLabels, self).__init__(cfg, model, optimizer)

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        logits = model(batch_data)
        return logits

    def configure_model(self, model: nn.Module):
        return model