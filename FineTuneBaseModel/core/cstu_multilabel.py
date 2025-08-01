import torch
import random
import math

class MemoryItem:
    def __init__(self, data, pseudo_label, uncertainty, age=0):
        self.data = data
        self.pseudo_label = pseudo_label
        self.uncertainty = uncertainty
        self.age = age

    def increase_age(self):
        self.age += 1

class CSTUMultiLabel:
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0):
        self.capacity = capacity
        self.num_class = num_class
        # Sức chứa mỗi ngăn vẫn được chia đều
        self.per_class_capacity = math.ceil(capacity / num_class) # Dùng ceil để đảm bảo không bị thiếu
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.data = {i: [] for i in range(self.num_class)}
        
        print(f"Initialized CSTUMultiLabel. Capacity: {self.capacity}, Per-Class Capacity: {self.per_class_capacity}")

    def get_unique_occupancy(self):
        """Đếm số lượng ảnh duy nhất."""
        return len({id(item.data) for class_list in self.data.values() for item in class_list})

    def get_occupancy(self):
        """Trả về một dict chứa số lượng mẫu trong mỗi ngăn."""
        # Tạm thời trả về tổng số ảnh duy nhất để hiển thị trên tqdm
        return self.get_unique_occupancy()

    def per_class_dist(self):
        return [len(self.data[c]) for c in range(self.num_class)]

    def add_instance(self, instance):
        x, pseudo_label, uncertainty = instance        
        # Tăng tuổi cho tất cả các mẫu hiện có TRƯỚC KHI thêm mẫu mới
        self.add_age()

        positive_classes = torch.where(pseudo_label > 0.1)[0]
        new_item = MemoryItem(data=x.cpu(), pseudo_label=pseudo_label.cpu(), uncertainty=uncertainty, age=0)
        new_score = self.heuristic_score(0, uncertainty)

        for cls_idx in positive_classes:
            cls_idx = cls_idx.item() # Chuyển tensor thành int
            # Logic quyết định và loại bỏ được gọi cho từng lớp
            if self.decide_and_remove(cls_idx, new_score):
                self.data[cls_idx].append(new_item)

    def decide_and_remove(self, cls_to_add, score_to_beat):
        """
        Logic quyết định có thêm mẫu vào ngăn 'cls_to_add' hay không.
        """
        class_list = self.data[cls_to_add]
        
        # Nếu ngăn của lớp này chưa đầy, luôn thêm vào
        if len(class_list) < self.per_class_capacity:
            return True
        else:
            # Nếu ngăn đã đầy, phải loại bỏ một mẫu từ chính ngăn đó
            # để nhường chỗ, với điều kiện mẫu mới "tốt hơn".
            return self.remove_from_classes([cls_to_add], score_to_beat)

    def remove_from_classes(self, classes_to_search, score_to_beat):
        worst_item_info = {'class': None, 'index': None, 'score': -1.0}

        for cls in classes_to_search:
            for idx, item in enumerate(self.data[cls]):
                score = self.heuristic_score(item.age, item.uncertainty)
                if score > worst_item_info['score']:
                    worst_item_info.update({'class': cls, 'index': idx, 'score': score})
        
        # Nếu tìm thấy một mẫu để loại bỏ VÀ điểm của nó tệ hơn (lớn hơn) điểm của mẫu mới
        if worst_item_info['class'] is not None and worst_item_info['score'] > score_to_beat:
            cls, idx = worst_item_info['class'], worst_item_info['index']
            self.data[cls].pop(idx)
            # print(f"remove_from_classes ->Worst: {worst_item_info['score']} compare to new {score_to_beat}")
            return True
            
        # Không thêm vào nếu không tìm thấy mẫu nào để thay thế
        # hoặc mẫu tệ nhất vẫn tốt hơn mẫu mới
        return False

    def get_majority_classes(self):
        # Hàm này có thể không cần thiết nữa với logic decide_and_remove mới,
        # nhưng vẫn giữ lại để có thể dùng trong tương lai.
        per_class_dist = self.per_class_dist()
        if not any(per_class_dist): return []
        max_occupied = max(per_class_dist)
        return [i for i, count in enumerate(per_class_dist) if count == max_occupied]

    def heuristic_score(self, age, uncertainty):
        age_score = self.lambda_t * (1 / (1 + math.exp(-age / (self.capacity + 1e-6))))
        uncertainty_score = self.lambda_u * uncertainty
        return age_score + uncertainty_score

    def add_age(self):
        # Dùng set để chỉ tăng tuổi cho mỗi item duy nhất một lần
        unique_items = {id(item.data): item for class_list in self.data.values() for item in class_list}.values()
        for item in unique_items:
            item.increase_age()

    def get_memory(self):
        unique_items = list({id(item.data): item for class_list in self.data.values() for item in class_list}.values())
        
        if not unique_items:
            return [], [], []

        tmp_data = [item.data for item in unique_items]
        tmp_labels = [item.pseudo_label for item in unique_items]
        tmp_age = [item.age for item in unique_items]

        return tmp_data, tmp_labels, tmp_age