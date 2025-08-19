import numpy as np
import torch
import torch.nn as nn
from ..utils import memory_multilabel as memory
from .base_adapter import BaseAdapter
from .base_adapter import bce_entropy
from ..utils.bn_layers import RobustBN1d, RobustBN2d
from ..utils.utils import set_named_submodule, get_named_submodule
from ..utils.custom_transforms import get_tta_transforms
from ..utils.constants import DEVICE
import wandb
from omegaconf import OmegaConf

class RoTTA_MultiLabels(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super(RoTTA_MultiLabels, self).__init__(cfg, model, optimizer)
        self.mem = memory.CSTU_MultiLabel(capacity=self.cfg.ADAPTER.RoTTA.MEMORY_SIZE, 
                                          num_class=cfg.MODEL.NUM_CLASSES, 
                                          lambda_t=cfg.ADAPTER.RoTTA.LAMBDA_T, 
                                          lambda_u=cfg.ADAPTER.RoTTA.LAMBDA_U)
        self.model_ema = self.build_ema(self.model)
        self.transform = get_tta_transforms(cfg)
        self.nu = cfg.ADAPTER.RoTTA.NU
        self.update_frequency = cfg.ADAPTER.RoTTA.UPDATE_FREQUENCY
        self.current_instance = 0
        self.labels_list = cfg.DATASET.LABELS_LIST

        target_indices_list = [0, 1, 2, 3, 4]
        self.target_indices = torch.tensor(target_indices_list, device=DEVICE)

         # Init wandb run
        cfg2 = OmegaConf.load("configs/adapter/rotta.yaml")
        wandb.init(
            project="chexpert-rotta",
            config=OmegaConf.to_container(cfg2, resolve=True), # Log toàn bộ config
            name=f"{cfg.MODEL.ARCH}-adapter{cfg.ADAPTER.NAME}-lr{cfg.ADAPTER.RoTTA.MEMORY_SIZE}-bs{cfg.TEST.BATCH_SIZE}"
        )
        wandb.watch(model, log="all", log_freq=100)

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        # batch data
        with torch.no_grad():
            model.train()
            self.model_ema.train()
            
            ema_out = self.model_ema(batch_data)
            predict_prob = torch.sigmoid(ema_out)
            pseudo_label = (predict_prob > 0.5).float() 
            
            # Compute uncertainty for Sigmoid outputs
            # (Sum of binary cross-entropy across classes)
            entropy = - (predict_prob * torch.log(predict_prob + 1e-6) + \
                        (1 - predict_prob) * torch.log(1 - predict_prob + 1e-6))
            entropy = torch.sum(entropy, dim=1)

        # add into memory
        for i, data in enumerate(batch_data):
            p_l = pseudo_label[i] 
            uncertainty = entropy[i].item()
            
            current_instance = (data, p_l, uncertainty)

            if p_l.sum() == 0:
                continue

            isAdded = self.mem.add_instance(current_instance)
            if isAdded: 
                self.current_instance += 1

            if self.current_instance % self.update_frequency == 0:
                self.update_model(model, optimizer)
        with torch.no_grad():
            final_logits = self.model_ema(batch_data)

        return final_logits

    def update_model(self, model, optimizer):
        model.train()
        self.model_ema.train()
        # get memory data
        sup_data, ages = self.mem.get_memory()
        l_sup = None
        if len(sup_data) > 0:
            sup_data = torch.stack(sup_data).to(DEVICE)
            ages = torch.tensor(ages).float().to(DEVICE)

            strong_sup_aug = self.transform(sup_data)
            
            # Disable gradient computation for the teacher model
            with torch.no_grad():
                ema_sup_out = self.model_ema(sup_data)
                
            stu_sup_out = model(strong_sup_aug)
            instance_weight = timeliness_reweighting(ages)
            
            l_sup = (bce_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()

        l = l_sup
        if l is not None:
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        self.update_ema_variables(self.model_ema, self.model, self.nu)
        
        stats = self.analyze_memory_bank()
        if stats:
            wandb.log(stats, step=self.current_instance)
    
    def analyze_memory_bank(self):
        all_items = self.mem.get_all_items()
        
        if not all_items:
            print("Memory bank is empty. No stats to analyze.")
            return None

        stats = {
            "memory/occupancy": self.mem.get_occupancy(),
        }

        # Class distribution in the memory
        class_dist = self.mem.per_class_dist()
        if hasattr(self, 'labels_list'):
            for i, class_name in enumerate(self.labels_list):
                stats[f"memory/dist/{class_name}"] = class_dist[i]
        else:
             for i, count in enumerate(class_dist):
                stats[f"memory/dist/class_{i}"] = count

        # Statistics on Uncertainty and Age
        uncertainties = [item.uncertainty for item in all_items]
        ages = [item.age for item in all_items]
        
        stats["memory/avg_uncertainty"] = np.mean(uncertainties) if uncertainties else 0
        stats["memory/max_uncertainty"] = np.max(uncertainties) if uncertainties else 0
        stats["memory/avg_age"] = np.mean(ages) if ages else 0
        stats["memory/max_age"] = np.max(ages) if ages else 0
        
        return stats
    
    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model

    def configure_model(self, model: nn.Module):

        model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = get_named_submodule(model, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = RobustBN2d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(bn_layer,
                                self.cfg.ADAPTER.RoTTA.ALPHA)
            momentum_bn.requires_grad_(True)
            set_named_submodule(model, name, momentum_bn)

        # --- Phần 2: Mở băng lớp Classifier (Bổ sung) ---
    
        classifier_found = False
        # Logic tìm kiếm đơn giản và mạnh mẽ hơn:
        # Tìm thuộc tính 'fc' hoặc 'classifier' và mở băng nó, BẤT KỂ nó là gì
        # (Linear, Sequential, hay một nn.Module tùy chỉnh khác).
        
        if hasattr(model, 'fc'):
            print("Found 'fc' attribute, making all its parameters trainable.")
            classifier_module = model.fc
            for param in classifier_module.parameters():
                param.requires_grad = True
            classifier_found = True
            
        elif hasattr(model, 'classifier'):
            print("Found 'classifier' attribute, making all its parameters trainable.")
            classifier_module = model.classifier
            for param in classifier_module.parameters():
                param.requires_grad = True
            classifier_found = True
            
        if not classifier_found:
            print("Warning: Could not find a standard classifier attribute ('fc' or 'classifier') to make trainable.")

        # In ra để xác nhận
        print("\n--- Trainable Parameters (ALL) ---")
        count = 0
        trainable_params_list = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params_list.append(name)
                count += 1
        print(f"Total number of trainable parameters: {count}")
        # print("Trainable parameters:", trainable_params_list) # Bỏ comment nếu muốn xem chi tiết
        print("--------------------------------\n")

        return model

def timeliness_reweighting(ages):
    return torch.exp(-ages) / (1 + torch.exp(-ages))