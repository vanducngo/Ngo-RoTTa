# RoTTA.py

import numpy as np
import torch
import torch.nn as nn
from ..utils import memory_multilabel as memory
from .base_adapter import BaseAdapter
from copy import deepcopy
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
        # SỬA ĐỔI: Sử dụng lớp Memory mới cho bài toán đa nhãn
        self.mem = memory.CSTU_MultiLabel(capacity=self.cfg.ADAPTER.RoTTA.MEMORY_SIZE, 
                                          num_class=cfg.MODEL.NUM_CLASSES, 
                                          lambda_t=cfg.ADAPTER.RoTTA.LAMBDA_T, 
                                          lambda_u=cfg.ADAPTER.RoTTA.LAMBDA_U)
        self.model_ema = self.build_ema(self.model)
        self.transform = get_tta_transforms(cfg)
        # self.transform = nn.Identity()
        self.nu = cfg.ADAPTER.RoTTA.NU
        self.update_frequency = cfg.ADAPTER.RoTTA.UPDATE_FREQUENCY
        self.current_instance = 0
        self.labels_list = cfg.DATASET.LABELS_LIST

        target_indices_list = [0, 1, 2, 3, 4]
        self.target_indices = torch.tensor(target_indices_list, device=DEVICE)

         # Khởi tạo wandb run

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
            model.eval()
            self.model_ema.eval()
            
            
            ema_out_14_cls = self.model_ema(batch_data)
            ema_out_5_cls = torch.index_select(ema_out_14_cls, 1, self.target_indices)
            predict_prob = torch.sigmoid(ema_out_5_cls)
            pseudo_label = (predict_prob > 0.5).float() 
            
            # Tính uncertainty cho đầu ra Sigmoid
            # (Tổng của binary cross-entropy trên mỗi lớp)
            entropy = - (predict_prob * torch.log(predict_prob + 1e-6) + \
                        (1 - predict_prob) * torch.log(1 - predict_prob + 1e-6))
            entropy = torch.sum(entropy, dim=1)

        # add into memory
        for i, data in enumerate(batch_data):
            # SỬA ĐỔI: pseudo_label giờ là một vector
            p_l = pseudo_label[i] 
            uncertainty = entropy[i].item()
            
            current_instance = (data, p_l, uncertainty)

            if p_l.sum() == 0:
                continue

            self.mem.add_instance(current_instance)
            self.current_instance += 1

            if self.current_instance % self.update_frequency == 0:
                self.update_model(model, optimizer)
                pass
        
        with torch.no_grad():
            final_logits_14_cls = self.model_ema(batch_data)
            final_logits_5_cls = torch.index_select(final_logits_14_cls, 1, self.target_indices)

        return final_logits_5_cls

    def update_model(self, model, optimizer):
        model.train()
        self.model_ema.train()
        # get memory data
        sup_data, ages = self.mem.get_memory()
        l_sup = None
        # print(f'len(sup_data) => {len(sup_data)}')
        if len(sup_data) > 0:
            sup_data = torch.stack(sup_data).to(DEVICE) # Chuyển lên device
            ages = torch.tensor(ages).float().to(DEVICE) # Chuyển lên device

            strong_sup_aug = self.transform(sup_data)
            
            # Tắt grad cho teacher model
            with torch.no_grad():
                ema_sup_out = self.model_ema(sup_data)
                
            stu_sup_out = model(strong_sup_aug)
            instance_weight = timeliness_reweighting(ages)
            
            # SỬA ĐỔI: Sử dụng loss consistency cho đa nhãn (BCE-based)
            l_sup = (bce_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()

        l = l_sup
        if l is not None:
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # print(f'Training student -> loss: {l}')

        self.update_ema_variables(self.model_ema, self.model, self.nu)
        
        stats = self.analyze_memory_bank()
        if stats:
            wandb.log(stats, step=self.current_instance)
    
    def analyze_memory_bank(self):
        """
        Tính toán và trả về các chỉ số thống kê của memory bank đa nhãn.
        """
        # Lấy tất cả các item từ memory bank
        all_items = self.mem.get_all_items()
        
        # Kiểm tra xem memory có rỗng không
        if not all_items:
            print("Memory bank is empty. No stats to analyze.")
            return None

        # 1. Các chỉ số cơ bản
        stats = {
            "memory/occupancy": self.mem.get_occupancy(),
            # Đối với CSTU_MultiLabel, không có khái niệm "unique" theo id(data) nữa
            # vì mỗi item là duy nhất. occupancy là đủ.
        }

        # 2. Phân phối lớp trong memory
        # Sử dụng hàm per_class_dist đã sửa đổi
        class_dist = self.mem.per_class_dist()
        # Giả sử self.labels_list được lưu trong adapter
        # Bạn có thể cần truyền nó vào từ config trong __init__
        if hasattr(self, 'labels_list'):
            for i, class_name in enumerate(self.labels_list):
                stats[f"memory/dist/{class_name}"] = class_dist[i]
        else:
             for i, count in enumerate(class_dist):
                stats[f"memory/dist/class_{i}"] = count

        # 3. Thống kê về Uncertainty và Age
        # Trích xuất dữ liệu từ danh sách các item
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
                print(f'Configure Model => Append BatchNorm1d and BatchNorm2d')
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = get_named_submodule(model, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                print(f'Replace  => Append BatchNorm1d to RobustBN1d')
                NewBN = RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                print(f'Replace  => Append BatchNorm2d to BatchNorm2d')
                NewBN = RobustBN2d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(bn_layer, self.cfg.ADAPTER.RoTTA.ALPHA)
            momentum_bn.requires_grad_(True)
            set_named_submodule(model, name, momentum_bn)
        return model

# timeliness_reweighting không thay đổi
def timeliness_reweighting(ages):
    # Đảm bảo ages đã ở trên đúng device
    return torch.exp(-ages) / (1 + torch.exp(-ages))