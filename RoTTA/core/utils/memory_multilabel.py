# memory_multilabel.py

import torch
import math

from core.utils.constants import IS_CPU_DEVICE

class MemoryItem:
    def __init__(self, data=None, label=None, uncertainty=0, age=0):
        self.data = data
        self.label = label # Thêm label để dễ truy cập
        self.uncertainty = uncertainty
        self.age = age

    def increase_age(self):
        self.age += 1

# Lớp CSTU mới cho bài toán đa nhãn
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
        # uncertainty là entropy (càng cao càng không chắc chắn)
        # age càng cao càng cũ
        # Chúng ta muốn tìm item có age LỚN và uncertainty LỚN
        
        # Chuẩn hóa age để có cùng thang đo với uncertainty
        # normalized_age = age / self.capacity
        normalized_age = 1 / (1 + math.exp(-age / self.capacity))
        
        # uncertainty/math.log(self.num_class) là từ code gốc
        # Giữ nguyên để có thang đo tương tự
        normalized_uncertainty = uncertainty / math.log(self.num_class) if self.num_class > 1 else uncertainty

        return self.lambda_t * normalized_age + self.lambda_u * normalized_uncertainty

    def add_age(self):
        for item in self.memory:
            item.increase_age()

    # Cung cấp truy cập trực tiếp vào danh sách các item
    def get_all_items(self) -> list[MemoryItem]:
        return self.memory
    
    def per_class_dist(self) -> list[int]:
        # Chuyển tensor về list trên CPU để xử lý bên ngoài
        class_counts = self._recalculate_class_counts()
        return class_counts.cpu().tolist()

    def get_memory(self):
        tmp_data = []
        tmp_age = []

        for item in self.memory:
            tmp_data.append(item.data)
            tmp_age.append(item.age)
            

        tmp_age = [x / self.capacity for x in tmp_age]
        # Chuẩn hóa age để dùng trong timeliness_reweighting
        # Chú ý: trong code gốc, age được chia cho capacity ở đây
        # nhưng timeliness_reweighting lại không dùng đến capacity.
        # Để nhất quán, ta truyền age chưa chuẩn hóa
        return tmp_data, tmp_age
    
    def add_instance(self, instance):
        """
        Thêm một mẫu mới vào memory bank đa nhãn.
        instance: tuple (data, prediction_vector, uncertainty_score)
        """
        assert len(instance) == 3
        
        x, prediction, uncertainty = instance
        new_item = MemoryItem(data=x, label=prediction, uncertainty=uncertainty, age=0)
        new_score = self.heuristic_score(0, uncertainty)

        if self.remove_instance(prediction, new_score):
            self.memory.append(new_item)

        self.add_age()

    def remove_instance(self, new_prediction, new_score) -> bool:
        # Bỏ qua các prediction không có phát hiện bệnh nào
        if new_prediction.sum() == 0:
            return False

        # Trường hợp 1: Memory chưa đầy, luôn thêm
        if self.get_occupancy() < self.capacity:
            return True

        # Trường hợp 2: Memory đã đầy, cần dọn chỗ
        current_counts = self._recalculate_class_counts()
        new_item_classes = torch.where(new_prediction > 0)[0]

        # Kiểm tra xem có lớp nào của item mới đang "thiếu" không
        # "Thiếu" nghĩa là số lượng hiện tại < ngưỡng mong muốn
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
        Tái tạo logic của `remove_from_classes`.
        Tìm item tệ nhất trong các lớp ứng cử viên và xóa nếu nó tệ hơn item mới.
        """
        max_score = -1.0
        replace_idx = -1

        for i, item in enumerate(self.memory):
            # Kiểm tra xem item này có thuộc bất kỳ lớp ứng cử viên nào không
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
                return True # Xóa thành công, có chỗ trống
            else:
                return False # Không đủ tệ để xóa
        else:
            # Không tìm thấy ứng cử viên nào để xóa (hiếm), không thể thêm
            # Logic gốc trả về True ở đây, nhưng False có vẻ an toàn hơn
            return False
        
    def get_majority_classes_indices(self, current_counts=None) -> list:
        if current_counts is None:
            current_counts = self._recalculate_class_counts()
            
        max_occupied = torch.max(current_counts).item()
        # Trả về một list các chỉ số của các lớp có số lượng bằng max_occupied
        return torch.where(current_counts == max_occupied)[0].tolist()
    

    """
        Tính toán lại và trả về số lượng mẫu cho mỗi lớp dựa trên trạng thái hiện tại của memory.
    """
    def _recalculate_class_counts(self) -> torch.Tensor: 
        if not self.memory:
            return torch.zeros(self.num_class, dtype=torch.long, device=self.device)

        # Lấy tất cả các vector label từ memory
        all_labels = [item.label for item in self.memory]
        
        # Stack chúng lại thành một tensor và tính tổng theo cột
        class_counts = torch.stack(all_labels).long().sum(dim=0)
        return class_counts