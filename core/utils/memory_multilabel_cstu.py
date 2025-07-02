import torch
import random
import copy
import math

class MemoryItem:
    def __init__(self, data=None, pseudo_label=None, uncertainty=0.0, age=0):
        # Lưu cả pseudo_label để có thể lấy lại khi cần
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
        self.per_class = self.capacity / self.num_class
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u

        # self.data là một dict, key là class_index, value là list các MemoryItem
        # Dùng dict để dễ truy cập hơn list of lists
        self.data = {i: [] for i in range(self.num_class)}
        
        # Thêm một set để theo dõi các tensor ảnh duy nhất đã được lưu, tránh trùng lặp
        self.unique_data_ids = set()

    def get_occupancy(self):
        # Tổng số slot đã được sử dụng (một ảnh có thể chiếm nhiều slot)
        return sum(len(class_list) for class_list in self.data.values())

    def per_class_dist(self):
        return [len(self.data[c]) for c in range(self.num_class)]

    def add_instance(self, instance):
        """
        instance: tuple (x, pseudo_label, uncertainty)
        pseudo_label là một tensor nhị phân [0, 1, 1, 0, ...]
        """
        assert (len(instance) == 3)
        x, pseudo_label, uncertainty = instance
        
        # Tăng tuổi cho tất cả các mẫu hiện có TRƯỚC KHI thêm mẫu mới
        self.add_age()
        
        # === THAY ĐỔI CỐT LÕI 1: LẶP QUA CÁC LỚP ===
        # Tìm các lớp mà mẫu này được dự đoán là dương tính
        positive_classes = torch.where(pseudo_label == 1)[0]
        
        new_item = MemoryItem(data=x.cpu(), pseudo_label=pseudo_label.cpu(), uncertainty=uncertainty, age=0)
        new_score = self.heuristic_score(0, uncertainty)

        # Lặp qua từng lớp dương tính và xem xét thêm mẫu vào ngăn của lớp đó
        for cls_idx in positive_classes:
            cls_idx = cls_idx.item() # Chuyển tensor thành int
            # Logic loại bỏ và thêm vào được gọi cho từng lớp
            if self.remove_instance(cls_idx, new_score):
                self.data[cls_idx].append(new_item)
                # Thêm id của tensor dữ liệu vào set để theo dõi
                self.unique_data_ids.add(id(new_item.data))

    def remove_instance(self, cls_to_add, score_to_beat):
        """
        Quyết định xem có nên thêm mẫu mới vào ngăn 'cls_to_add' hay không.
        Nếu cần, nó sẽ loại bỏ một mẫu cũ từ một nơi nào đó.
        """
        class_list = self.data[cls_to_add]
        
        # Nếu ngăn của lớp này chưa đầy, ưu tiên thêm vào
        if len(class_list) < self.per_class:
            return True
        # Nếu ngăn đã đầy, nhưng tổng thể memory chưa đầy, vẫn có thể thêm
        elif self.get_occupancy() < self.capacity:
            return True
        else:
            # Memory đã đầy, cần loại bỏ một mẫu nào đó
            # Ưu tiên loại bỏ từ chính ngăn này
            return self.remove_from_classes([cls_to_add], score_to_beat)

    def remove_from_classes(self, classes_to_search: list[int], score_to_beat):
        """
        Tìm và loại bỏ mẫu có điểm heuristic cao nhất từ các ngăn được chỉ định.
        """
        worst_item_info = {'class': None, 'index': None, 'score': -1}

        for cls in classes_to_search:
            for idx, item in enumerate(self.data[cls]):
                score = self.heuristic_score(item.age, item.uncertainty)
                if score > worst_item_info['score']:
                    worst_item_info.update({'class': cls, 'index': idx, 'score': score})
        
        # Nếu tìm thấy một mẫu để loại bỏ VÀ điểm của nó tệ hơn mẫu mới
        if worst_item_info['class'] is not None and worst_item_info['score'] > score_to_beat:
            cls, idx = worst_item_info['class'], worst_item_info['index']
            # Trước khi pop, kiểm tra xem có nơi nào khác lưu trữ tensor này không
            item_to_remove = self.data[cls][idx]
            self.data[cls].pop(idx)
            
            # Kiểm tra xem tensor dữ liệu có còn tồn tại ở đâu khác không
            is_still_present = any(id(item.data) == id(item_to_remove.data) for lst in self.data.values() for item in lst)
            if not is_still_present and id(item_to_remove.data) in self.unique_data_ids:
                self.unique_data_ids.remove(id(item_to_remove.data))

            return True
        else:
            # Không tìm thấy mẫu nào để loại bỏ, hoặc mẫu tệ nhất vẫn tốt hơn mẫu mới
            return False

    def get_majority_classes(self):
        # Hàm này có thể không cần thiết nữa nếu logic remove_instance được đơn giản hóa,
        # nhưng vẫn giữ lại để tương thích.
        per_class_dist = self.per_class_dist()
        max_occupied = max(per_class_dist) if per_class_dist else 0
        return [i for i, count in enumerate(per_class_dist) if count == max_occupied]

    def heuristic_score(self, age, uncertainty):
        # === THAY ĐỔI CỐT LÕI 2: ĐIỀU CHỈNH CÔNG THỨC ===
        # Giả định uncertainty đã được chuẩn hóa trong khoảng [0, 1]
        # Bỏ mẫu số math.log(self.num_class)
        age_score = self.lambda_t * (1 / (1 + math.exp(-age / (self.capacity + 1e-6))))
        uncertainty_score = self.lambda_u * uncertainty
        return age_score + uncertainty_score

    def add_age(self):
        for class_list in self.data.values():
            for item in class_list:
                item.increase_age()

    def get_memory(self):
        # === THAY ĐỔI CỐT LÕI 3: LẤY MẪU DUY NHẤT ===
        # Tránh việc một ảnh được lấy nhiều lần trong một batch
        
        # Tạo một dict để lưu trữ các mẫu duy nhất, key là id của tensor data
        unique_samples = {id(item.data): item for class_list in self.data.values() for item in class_list}
        
        all_items = list(unique_samples.values())
        
        if not all_items:
            return [], [], []

        tmp_data = [item.data for item in all_items]
        tmp_labels = [item.pseudo_label for item in all_items]
        tmp_age = [item.age for item in all_items]

        # Chuẩn hóa tuổi
        tmp_age = [age / (self.capacity + 1e-6) for age in tmp_age]

        return tmp_data, tmp_labels, tmp_age
    
    def get_list_class_name(self):
        """Trả về số lượng mẫu trong mỗi ngăn."""
        return self.num_class