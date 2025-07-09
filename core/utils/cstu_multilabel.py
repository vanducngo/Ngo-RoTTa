# File: core/utils/cstu_multilabel.py
import torch
import random
import math

class MemoryItem:
    def __init__(self, data, pseudo_label, uncertainty, age=0):
        self.data = data
        self.pseudo_label = pseudo_label # Sửa đổi: Lưu toàn bộ nhãn giả
        self.uncertainty = uncertainty
        self.age = age

    def increase_age(self):
        self.age += 1

class CSTUMultiLabel:
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0):
        self.capacity = capacity
        self.num_class = num_class
        self.per_class_capacity = self.capacity / self.num_class
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.data = {i: [] for i in range(self.num_class)}
        self.unique_item_ids = set() # Dùng để theo dõi các item duy nhất

    def get_occupancy(self):
        return sum(len(class_list) for class_list in self.data.values())

    def per_class_dist(self):
        return [len(self.data[c]) for c in range(self.num_class)]

    def add_instance(self, instance):
        x, pseudo_label, uncertainty = instance
        self.add_age()

        new_item = MemoryItem(data=x.cpu(), pseudo_label=pseudo_label.cpu(), uncertainty=uncertainty, age=0)
        new_score = self.heuristic_score(0, uncertainty)

        # THAY ĐỔI 1: Lặp qua các lớp dương tính
        positive_classes = torch.where(pseudo_label == 1)[0]
        for cls_idx_tensor in positive_classes:
            cls_idx = cls_idx_tensor.item()
            if self.remove_instance(cls_idx, new_score):
                self.data[cls_idx].append(new_item)
                self.unique_item_ids.add(id(new_item))

    def remove_instance(self, cls_to_add, score_to_beat):
        class_list = self.data[cls_to_add]
        
        # Logic được giữ nguyên, nhưng giờ áp dụng cho từng ngăn
        if len(class_list) < self.per_class_capacity:
            return True
        elif self.get_occupancy() < self.capacity:
            return True
        else:
            # Ưu tiên loại bỏ từ các lớp đầy nhất (majority classes)
            majority_classes = self.get_majority_classes()
            return self.remove_from_classes(majority_classes, score_to_beat)
            
    def remove_from_classes(self, classes_to_search, score_to_beat):
        worst_item_info = {'class': None, 'index': None, 'score': -1.0}

        for cls in classes_to_search:
            for idx, item in enumerate(self.data[cls]):
                score = self.heuristic_score(item.age, item.uncertainty)
                if score > worst_item_info['score']:
                    worst_item_info.update({'class': cls, 'index': idx, 'score': score})
        
        if worst_item_info['class'] is not None and worst_item_info['score'] > score_to_beat:
            cls, idx = worst_item_info['class'], worst_item_info['index']
            item_to_remove = self.data[cls].pop(idx)
            # Nếu item này không còn ở bất kỳ ngăn nào khác, xóa khỏi set duy nhất
            if id(item_to_remove) in self.unique_item_ids:
                is_still_present = any(id(item) == id(item_to_remove) for lst in self.data.values() for item in lst)
                if not is_still_present:
                    self.unique_item_ids.remove(id(item_to_remove))
            return True
        return False

    def get_majority_classes(self):
        per_class_dist = self.per_class_dist()
        if not per_class_dist: return []
        max_occupied = max(per_class_dist)
        return [i for i, count in enumerate(per_class_dist) if count == max_occupied]

    def heuristic_score(self, age, uncertainty):
        # THAY ĐỔI 2: Đơn giản hóa điểm uncertainty cho đa nhãn
        age_score = self.lambda_t * (1 / (1 + math.exp(-age / (self.capacity + 1e-6))))
        uncertainty_score = self.lambda_u * uncertainty
        return age_score + uncertainty_score

    def add_age(self):
        for class_list in self.data.values():
            for item in class_list:
                item.increase_age()

    def get_memory(self):
        # THAY ĐỔI 3: Lấy các mẫu duy nhất để huấn luyện
        unique_items = {id(item): item for class_list in self.data.values() for item in class_list}.values()
        
        if not unique_items:
            return [], [], []

        tmp_data = [item.data for item in unique_items]
        tmp_labels = [item.pseudo_label for item in unique_items]
        tmp_age = [item.age for item in unique_items]

        return tmp_data, tmp_labels, tmp_age