import torch
import torch.nn as nn
import torch.optim as optim
import copy
import logging
import wandb
from omegaconf import OmegaConf
import random
import numpy as np

from constants import COMMON_FINAL_LABEL_SET, TARGET_INDICES_IN_FULL_LIST

from .base_adapter import BaseAdapter
from .cstu_multilabel import CSTUMultiLabel 

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import logging
import random
import wandb
from omegaconf import OmegaConf
import numpy as np

# def timeliness_reweighting(ages, device):
#     """Hàm helper để tính trọng số dựa trên tuổi của mẫu."""
#     if not isinstance(ages, torch.Tensor):
#         ages = torch.tensor(ages, dtype=torch.float32, device=device)
#     # Hàm sigmoid ngược để trọng số giảm khi tuổi tăng
#     return torch.exp(-ages / 100.0)

# class RoTTAMultiLabelConsistent(BaseAdapter):
#     """
#     Phiên bản RoTTA cho đa nhãn, tích hợp các chiến lược nhất quán:
#     1. Thích ứng có chọn lọc trên mô hình 14 lớp.
#     2. Chỉ cập nhật các lớp BatchNorm và Classifier.
#     3. Sử dụng soft-labels từ Teacher.
#     4. Áp dụng Confidence Thresholding để chỉ học từ các nhãn giả đáng tin cậy.
#     """
#     def __init__(self, cfg, model, optimizer_func):
#         # Khởi tạo các thuộc tính cần thiết trước khi gọi super()
#         self.logger = logging.getLogger("TTA.adapter")
#         self.cfg = cfg
        
#         # Gọi __init__ của BaseAdapter
#         super().__init__(cfg, model, optimizer_func)

#         # Khởi tạo các thành phần của RoTTA
#         self.student = self.model
#         self.teacher = self.build_ema(self.student)
        
#         self.mem = CSTUMultiLabel(
#             capacity=self.cfg.ADAPTER.MEMORY_SIZE,
#             num_class=len(COMMON_FINAL_LABEL_SET),
#             lambda_t=self.cfg.ADAPTER.LAMBDA_T,
#             lambda_u=self.cfg.ADAPTER.LAMBDA_U
#         )
        
#         self.transform = nn.Identity()
#         self.nu = 1.0 - self.cfg.ADAPTER.EMA_DECAY
#         self.criterion = nn.BCEWithLogitsLoss(reduction='none') # Rất quan trọng: reduction='none'
#         self.update_frequency = self.cfg.ADAPTER.UPDATE_FREQUENCY
#         self.instance_counter = 0

#         device = next(self.student.parameters()).device
#         self.target_indices = torch.tensor(TARGET_INDICES_IN_FULL_LIST, device=device)

#         # Khởi tạo wandb run (an toàn hơn)
        
#         try:
#             if wandb.run is None:
#                 wandb.init(
#                     project="chexpert-rotta",
#                     config=OmegaConf.to_container(cfg, resolve=True), # Log toàn bộ config
#                     name=f"{cfg.MODEL.ARCH}-lr{cfg.TRAINING.LEARNING_RATE}-bs{cfg.TRAINING.BATCH_SIZE}"
#                 )
#                 wandb.watch(model, log="all", log_freq=100)
#             wandb.watch(self.student, log="all", log_freq=100)
#         except Exception as e:
#             self.logger.warning(f"Could not initialize wandb: {e}")

#     def configure_model(self, model: nn.Module):
#         """Cấu hình các tham số có thể huấn luyện."""
#         model.requires_grad_(False)
#         self.logger.info("Configuring model: Making BatchNorm layers and final classifier trainable.")
        
#         # Mở băng các lớp BatchNorm
#         for name, m in model.named_modules():
#             if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
#                 for param in m.parameters():
#                     param.requires_grad = True
        
#         # Mở băng lớp classifier cuối cùng
#         classifier = model.fc if hasattr(model, 'fc') else model.classifier
#         if classifier is not None:
#             for param in classifier.parameters():
#                 param.requires_grad = True
        
#         # In ra các tham số có thể huấn luyện để kiểm tra
#         self.logger.info("Trainable parameters:")
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 self.logger.info(f"  - {name}")
                
#         return model

#     @torch.no_grad()
#     def get_teacher_output_selective(self, x):
#         """Lấy đầu ra từ teacher, lọc ra 6 lớp để tạo nhãn giả và uncertainty."""
#         self.teacher.eval()
#         teacher_logits_14_cls = self.teacher(x)
#         teacher_logits_6_cls = torch.index_select(teacher_logits_14_cls, 1, self.target_indices)

#         probs = torch.sigmoid(teacher_logits_6_cls)
#         pseudo_labels_hard = (probs > 0.5).float()
#         uncertainties = torch.mean(1 - torch.abs(probs - 0.5) * 2, dim=1)
        
#         return pseudo_labels_hard, uncertainties

#     @torch.enable_grad()
#     def forward_and_adapt(self, x, model, optimizer):
#         pseudo_labels, mean_uncertainties = self.get_teacher_output_selective(x)

#         for i in range(x.size(0)):
#             instance = (x[i], pseudo_labels[i], mean_uncertainties[i].item())
#             self.mem.add_instance(instance)
#             self.instance_counter += 1
        
#         # Log lên wandb (nếu có)
#         if wandb.run:
#             self.analyze_and_log_wandb()

#         if self.instance_counter % self.update_frequency == 0:
#             bank_data, _, bank_ages = self.mem.get_memory()
            
#             if len(bank_data) >= self.cfg.ADAPTER.BATCH_SIZE:
#                 model.train()
                
#                 indices = random.sample(range(len(bank_data)), self.cfg.ADAPTER.BATCH_SIZE)
                
#                 batch_images = torch.stack([bank_data[i] for i in indices]).to(x.device)
#                 batch_ages_list = [bank_ages[i] for i in indices]

#                 strong_aug_images = self.transform(batch_images)
#                 student_logits_14_cls = model(strong_aug_images)

#                 with torch.no_grad():
#                     teacher_logits_14_cls = self.teacher(batch_images)

#                 student_logits_6_cls = torch.index_select(student_logits_14_cls, 1, self.target_indices)
#                 teacher_logits_6_cls = torch.index_select(teacher_logits_14_cls, 1, self.target_indices)

#                 soft_targets = torch.sigmoid(teacher_logits_6_cls)
                
#                 # --- ÁP DỤNG CONFIDENCE THRESHOLDING ---
#                 high_conf_thresh = self.cfg.ADAPTER.HIGH_CONF_THRESHOLD
#                 low_conf_thresh = 1.0 - high_conf_thresh
                
#                 high_conf_mask = (soft_targets > high_conf_thresh) | (soft_targets < low_conf_thresh)
                
#                 loss_all_positions = self.criterion(student_logits_6_cls, soft_targets)
#                 loss_high_conf = loss_all_positions * high_conf_mask
                
#                 num_high_conf = high_conf_mask.sum()

#                 if num_high_conf > 0:
#                     instance_loss = loss_high_conf.sum() / num_high_conf
#                     final_loss = instance_loss # Tạm thời chưa dùng timeliness_reweighting

#                     if optimizer is not None and final_loss > 0:
#                         self.logger.info(f"Updating model. Loss: {final_loss.item():.4f}. High-conf predictions: {num_high_conf.item()}/{loss_high_conf.numel()}")                        
#                         print(f"Updating model. Loss: {final_loss.item():.4f}. High-conf predictions: {num_high_conf.item()}/{loss_high_conf.numel()}")
                        
#                         optimizer.zero_grad()
#                         final_loss.backward()
#                         optimizer.step()
                        
#                         self.update_ema_variables(self.teacher, self.student, self.nu)

#         with torch.no_grad():
#             self.teacher.eval()
#             final_output_14_cls = self.teacher(x)
#             final_output_6_cls = torch.index_select(final_output_14_cls, 1, self.target_indices)
            
#         return final_output_6_cls
        
#     def analyze_and_log_wandb(self):
#         """Tính toán và trả về các chỉ số thống kê của memory bank."""
#         unique_items = list({id(item.data): item for class_list in self.mem.data.values() for item in class_list}.values())
#         if not unique_items:
#             return None

#         # 1. Các chỉ số cơ bản
#         stats = {
#             "memory/unique_occupancy": len(unique_items),
#             "memory/total_slots_used": self.mem.get_occupancy(),
#         }

#         # 2. Phân phối lớp trong memory
#         class_dist = self.mem.per_class_dist()
#         for i, class_name in enumerate(COMMON_FINAL_LABEL_SET):
#             stats[f"memory/dist/{class_name}"] = class_dist[i]

#         # 3. Thống kê về Uncertainty và Age
#         uncertainties = [item.uncertainty for item in unique_items]
#         ages = [item.age for item in unique_items]
#         stats["memory/avg_uncertainty"] = np.mean(uncertainties)
#         stats["memory/max_uncertainty"] = np.max(uncertainties)
#         stats["memory/avg_age"] = np.mean(ages)
#         stats["memory/max_age"] = np.max(ages)
        
#         return stats

def timeliness_reweighting(ages, device):
    if not isinstance(ages, torch.Tensor):
        ages = torch.tensor(ages, dtype=torch.float32, device=device)
    return torch.exp(-ages / 100.0)

class RoTTAMultiLabelConsistent(BaseAdapter):
    def __init__(self, cfg, model, optimizer_func):
        self.logger = logging.getLogger("TTA.adapter")
        self.cfg = cfg
        super().__init__(cfg, model, optimizer_func)

        # Khởi tạo các thành phần của RoTTA
        self.student = self.model
        self.teacher = self.build_ema(self.student)
        
        self.mem = CSTUMultiLabel(
            capacity=self.cfg.ADAPTER.MEMORY_SIZE,
            num_class=len(COMMON_FINAL_LABEL_SET),
            lambda_t=self.cfg.ADAPTER.LAMBDA_T,
            lambda_u=self.cfg.ADAPTER.LAMBDA_U
        )
        
        self.transform = nn.Identity() # Tạm thời chưa dùng strong augmentation
        self.nu = 1.0 - self.cfg.ADAPTER.EMA_DECAY
        # Dùng BCEWithLogitsLoss để so sánh logits của student và soft-labels (xác suất) của teacher
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.update_frequency = self.cfg.ADAPTER.UPDATE_FREQUENCY
        self.instance_counter = 0

        # Lưu lại các chỉ số của 6 lớp mục tiêu để lọc
        device = next(self.student.parameters()).device # Lấy device từ chính model
        self.target_indices = torch.tensor(TARGET_INDICES_IN_FULL_LIST, device=device)

        # Khởi tạo wandb run
        wandb.init(
            project="chexpert-rotta",
            config=OmegaConf.to_container(cfg, resolve=True), # Log toàn bộ config
            name=f"{cfg.MODEL.ARCH}-lr{cfg.TRAINING.LEARNING_RATE}-bs{cfg.TRAINING.BATCH_SIZE}"
        )
        wandb.watch(model, log="all", log_freq=100)

    def configure_model(self, model: nn.Module):
        model.requires_grad_(False)
        
        self.logger.info("Configuring model: Making BatchNorm layers and final classifier trainable.")
        trainable_param_names = []

        # Mở băng các lớp BatchNorm
        for name, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                # Mở băng toàn bộ lớp BN để an toàn
                for param in m.parameters():
                    param.requires_grad = True
                trainable_param_names.append(name)
        
        # Mở băng lớp classifier cuối cùng
        # Logic này cần phải mạnh mẽ để xử lý các kiến trúc khác nhau
        classifier = None
        if hasattr(model, 'fc'):
            classifier = model.fc
        elif hasattr(model, 'classifier'):
            classifier = model.classifier
        
        if classifier is not None:
            for param in classifier.parameters():
                param.requires_grad = True
            # Lấy tên của các tham số có thể huấn luyện trong classifier
            for name, _ in classifier.named_parameters():
                trainable_param_names.append(f"classifier.{name}") # Giả định tên
        
        self.logger.info(f"Trainable modules/layers set: {list(set(trainable_param_names))}")
        return model

    @torch.no_grad()
    def get_teacher_output_selective(self, x):
        """
        Lấy đầu ra từ teacher, lọc ra 6 lớp để tạo nhãn giả và uncertainty.
        """
        self.teacher.eval()
        teacher_logits_14_cls = self.teacher(x)
        
        teacher_logits_6_cls = torch.index_select(teacher_logits_14_cls, 1, self.target_indices)

        probs = torch.sigmoid(teacher_logits_6_cls)
        pseudo_labels_hard = (probs > 0.9).float()
        
        uncertainties = torch.mean(1 - torch.abs(probs - 0.5) * 2, dim=1)
        return pseudo_labels_hard, uncertainties

    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        # 1. Lấy nhãn giả và uncertainty từ teacher
        pseudo_labels, mean_uncertainties = self.get_teacher_output_selective(x)

        # 2. Cập nhật Memory Bank
        for i in range(x.size(0)):
            instance = (x[i], pseudo_labels[i], mean_uncertainties[i].item())
            self.mem.add_instance(instance)
            self.instance_counter += 1

        # === LOGIC HUẤN LUYỆN ĐÃ ĐƯỢC THIẾT KẾ LẠI HOÀN TOÀN ===
        
        # 3. Kích hoạt huấn luyện theo tần suất
        if self.instance_counter % self.cfg.ADAPTER.UPDATE_FREQUENCY == 0:
            print(f"Model train")
            model.train()
            
            # 3.1 Lấy toàn bộ các mẫu duy nhất hiện có trong memory
            unique_items = list({id(item.data): item for class_list in self.mem.data.values() for item in class_list}.values())
            
            # 3.2 KIỂM TRA ĐIỀU KIỆN MỚI: Chỉ cần có ít nhất một mẫu là có thể học
            if len(unique_items) >= self.cfg.ADAPTER.BATCH_SIZE:
                # 3.3 TẠO BATCH HUẤN LUYỆN
                # Nếu số mẫu ít hơn batch_size, thì học trên tất cả các mẫu đó.
                # Nếu nhiều hơn, thì lấy ngẫu nhiên một batch.
                current_batch_size = min(len(unique_items), self.cfg.ADAPTER.BATCH_SIZE)
                # batch_samples = random.sample(unique_items, current_batch_size)
                batch_samples = unique_items
                
                # Tạo tensor từ batch
                batch_images = torch.stack([s.data for s in batch_samples]).to(x.device)
                batch_labels = torch.stack([s.pseudo_label for s in batch_samples]).to(x.device)
                batch_ages = [s.age for s in batch_samples]

                # 3.4 THỰC HIỆN CẬP NHẬT
                student_logits_14_cls = model(batch_images)
                teacher_logits_14_cls = self.teacher(batch_images) # Cần teacher output cho soft labels

                student_logits_6_cls = torch.index_select(student_logits_14_cls, 1, self.target_indices)
                teacher_logits_6_cls = torch.index_select(teacher_logits_14_cls, 1, self.target_indices)

                soft_targets = torch.sigmoid(teacher_logits_6_cls)
                instance_loss = self.criterion(student_logits_6_cls, soft_targets).mean(dim=1)
                
                instance_weight = timeliness_reweighting(batch_ages, device=x.device)
                final_loss = (instance_loss * instance_weight).mean()
                
                print(f'training model -> Loss: {final_loss}')
                if optimizer is not None and final_loss > 0:
                    self.logger.info(f"Updating model with a batch of {current_batch_size}. Loss: {final_loss.item():.4f}")
                    optimizer.zero_grad()
                    final_loss.backward()
                    optimizer.step()
                
                self.update_ema_variables(self.teacher, self.student, self.nu)

                stats = self.analyze_memory_bank()
                if stats:
                    wandb.log(stats, step=self.instance_counter)

        # 4. Trả về kết quả để đánh giá
        with torch.no_grad():
            self.teacher.eval()
            final_output_14_cls = self.teacher(x)
            final_output_6_cls = torch.index_select(final_output_14_cls, 1, self.target_indices)
            
        return final_output_6_cls
    
    def analyze_memory_bank(self):
        """Tính toán và trả về các chỉ số thống kê của memory bank."""
        unique_items = list({id(item.data): item for class_list in self.mem.data.values() for item in class_list}.values())
        if not unique_items:
            return None

        # 1. Các chỉ số cơ bản
        stats = {
            "memory/unique_occupancy": len(unique_items),
            "memory/total_slots_used": self.mem.get_occupancy(),
        }

        # 2. Phân phối lớp trong memory
        class_dist = self.mem.per_class_dist()
        for i, class_name in enumerate(COMMON_FINAL_LABEL_SET):
            stats[f"memory/dist/{class_name}"] = class_dist[i]

        # 3. Thống kê về Uncertainty và Age
        uncertainties = [item.uncertainty for item in unique_items]
        ages = [item.age for item in unique_items]
        stats["memory/avg_uncertainty"] = np.mean(uncertainties)
        stats["memory/max_uncertainty"] = np.max(uncertainties)
        stats["memory/avg_age"] = np.mean(ages)
        stats["memory/max_age"] = np.max(ages)
        
        return stats

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(1.0 - nu).add_(param.data, alpha=nu)