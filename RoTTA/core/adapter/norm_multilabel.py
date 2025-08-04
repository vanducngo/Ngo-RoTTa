import torch.nn as nn
from .base_adapter import BaseAdapter

class NormMultiLabel(BaseAdapter):
    """
    Test-Time Normalization (NORM) for Multi-Label Classification.
    
    This adapter adapts a model by using batch-wise statistics for normalization
    during testing, instead of the running stats learned during training.
    It does not perform any gradient-based optimization.
    """
    def __init__(self, cfg, model, optimizer_func):
        # NORM không cần optimizer, nhưng chúng ta vẫn tuân theo cấu trúc của BaseAdapter.
        # optimizer_func sẽ không được sử dụng vì collect_params trả về list rỗng.
        super().__init__(cfg, model, optimizer_func)
        self.logger.info("NormMultiLabel adapter initialized.")

    def configure_model(self, model: nn.Module):
        """
        Configure the model for Test-Time Normalization.
        The key is to set normalization layers to train() mode, which makes them
        use the statistics of the current batch for normalization.
        """
        self.logger.info("Configuring model for NORM: Setting normalization layers to train mode.")
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
                # Setting to train() forces the layer to use batch statistics
                m.train()
        return model

    def collect_params(self, model: nn.Module):
        """
        NORM does not update any parameters via backpropagation.
        Therefore, we return empty lists.
        """
        return [], []

    def forward_and_adapt(self, x, model, optimizer):
        """
        For NORM, the "adaptation" happens implicitly during the forward pass
        within the normalization layers. There is no explicit adaptation step.
        
        Args:
            x (torch.Tensor): The input batch.
            model (nn.Module): The model configured for NORM.
            optimizer: This will be None and is not used.

        Returns:
            torch.Tensor: The model's output logits.
        """
        return model(x)