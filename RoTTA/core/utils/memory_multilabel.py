# memory_multilabel.py

import numpy as np
import torch
import math

class MemoryItem:
    def __init__(self, data=None, label=None, uncertainty=0, age=0):
        self.data = data
        self.label = label # Thêm label để dễ truy cập
        self.uncertainty = uncertainty
        self.age = age

    def increase_age(self):
        self.age += 1

class CSTU_MultiLabel:
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0):
        self.capacity = capacity
        self.num_class = num_class
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.desired_count_per_class = capacity / num_class

        self.memory: list[MemoryItem] = []

    def get_occupancy(self):
        return len(self.memory)

    def heuristic_score(self, age, uncertainty):
        # Uncertainty is measured by entropy (higher = more uncertain)
        # Age increases over time (higher = older)
        # We aim to find items with HIGH age and HIGH uncertainty

        # Normalize age to match the scale of uncertainty
        # normalized_age = age / self.capacity
        normalized_age = 1 / (1 + math.exp(-age / self.capacity))
        
        normalized_uncertainty = uncertainty / math.log(self.num_class) if self.num_class > 1 else uncertainty

        return self.lambda_t * normalized_age + self.lambda_u * normalized_uncertainty

    def add_age(self):
        for item in self.memory:
            item.increase_age()

    def get_all_items(self) -> list[MemoryItem]:
        return self.memory
    
    def per_class_dist(self) -> list[int]:
        class_counts = self._recalculate_class_counts()
        return class_counts.cpu().tolist()

    def get_memory(self):
        tmp_data = []
        tmp_age = []

        for item in self.memory:
            tmp_data.append(item.data)
            tmp_age.append(item.age)
            

        tmp_age = [x / self.capacity for x in tmp_age]
        return tmp_data, tmp_age
    
    def add_instance(self, instance):
        isAdded = False
        assert len(instance) == 3
        
        x, prediction, uncertainty = instance
        new_item = MemoryItem(data=x, label=prediction, uncertainty=uncertainty, age=0)
        new_score = self.heuristic_score(0, uncertainty)

        if self.remove_instance(prediction, new_score):
            isAdded = True
            self.memory.append(new_item)

        self.add_age()
        return isAdded

    def remove_instance(self, new_prediction, new_score) -> bool:
        if new_prediction.sum() == 0:
            return False

        # Case 1: Memory is not full, always add the item
        if self.get_occupancy() < self.capacity:
            return True

        # Case 2: Memory is full, need to make space
        current_counts = self._recalculate_class_counts()
        new_item_classes = torch.where(new_prediction > 0)[0]

        # Check if the new item's classes are "underrepresented"
        # "Underrepresented" means the current count is below the desired threshold
        is_under_represented = False
        for cls_idx in new_item_classes:
            if current_counts[cls_idx] < self.desired_count_per_class:
                is_under_represented = True
                break

        if is_under_represented:
            majority_classes_indices = self.get_majority_classes_indices(current_counts)
            return self.remove_from_classes(majority_classes_indices, new_score)
        else:
            return self.remove_from_classes(new_item_classes.tolist(), new_score)

    def remove_from_classes(self, candidate_classes: list, score_base) -> bool:
        """
        Reproduce the logic of `remove_from_classes`.
        Find the worst item among candidate classes and remove it if it's worse than the new item.
        """
        max_score = -1.0
        replace_idx = -1

        for i, item in enumerate(self.memory):
            is_candidate = False
            for cls_idx in candidate_classes:
                if item.label[cls_idx] > 0:
                    is_candidate = True
                    break
            
            if is_candidate:
                score = self.heuristic_score(item.age, item.uncertainty)
                if score > max_score:
                    max_score = score
                    replace_idx = i

        if replace_idx != -1:
            if max_score > score_base:
                self.memory.pop(replace_idx)
                return True 
            else:
                return False
        else:
            return False
        
    def get_majority_classes_indices(self, current_counts=None) -> list:
        if current_counts is None:
            current_counts = self._recalculate_class_counts()
            
        max_occupied = torch.max(current_counts).item()
        return torch.where(current_counts == max_occupied)[0].tolist()
    
    def get_majority_classes_indices_v2(self, current_counts=None) -> list:
        if current_counts is None:
            current_counts = self._recalculate_class_counts()
            
        # 1. Tính ngưỡng trung bình mong muốn
        # desired_count_per_class đã được định nghĩa trong __init__
        # desired_count = self.desired_count_per_class
        desired_count = 0

        # 2. Tìm tất cả các lớp có số lượng vượt ngưỡng
        # Chuyển sang numpy để thao tác dễ hơn
        current_counts_np = current_counts.cpu().numpy()
        majority_indices = np.where(current_counts_np >= desired_count)[0]
        
        # 3. Xử lý trường hợp biên: nếu không có lớp nào vượt ngưỡng
        if len(majority_indices) == 0:
            # Quay lại chiến lược cũ: tìm lớp có số lượng lớn nhất
            max_occupied = np.max(current_counts_np)
            # Có thể có nhiều lớp cùng có số lượng lớn nhất
            majority_indices = np.where(current_counts_np == max_occupied)[0]
        
        return majority_indices.tolist()

    def _recalculate_class_counts(self) -> torch.Tensor: 
        if not self.memory:
            return torch.zeros(self.num_class, dtype=torch.long, device=self.device)

        all_labels = [item.label for item in self.memory]
        class_counts = torch.stack(all_labels).long().sum(dim=0)
        return class_counts