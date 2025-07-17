# memory_multilabel.py

import random
import copy
import torch
import torch.nn.functional as F
import numpy as np
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

        # Sửa đổi: Chỉ dùng một list duy nhất để lưu các MemoryItem
        self.memory: list[MemoryItem] = []
        # Sửa đổi: Theo dõi số lần xuất hiện của mỗi lớp
        if IS_CPU_DEVICE:
            self.class_counts = torch.zeros(num_class, dtype=torch.long).cpu()
        else:
            self.class_counts = torch.zeros(num_class, dtype=torch.long).cuda()

    def get_occupancy(self):
        return len(self.memory)

    def add_instance(self, instance):
        assert len(instance) == 3
        x, prediction, uncertainty = instance # prediction giờ là một vector [0,1,0,1,...]
        new_item = MemoryItem(data=x, label=prediction, uncertainty=uncertainty, age=0)
        
        # Tăng tuổi các item cũ
        self.add_age()
        
        # Nếu memory chưa đầy, thêm trực tiếp
        if self.get_occupancy() < self.capacity:
            self.memory.append(new_item)
            # Cập nhật số đếm lớp
            self.class_counts += prediction.long()
        else: # Nếu memory đã đầy, cần thay thế
            
            ####### Phiên bản ban đầu #######
            # # Tìm lớp chiếm ưu thế nhất trong số các lớp của instance mới
            # # Chỉ xét các lớp mà instance này thuộc về (prediction > 0)
            # instance_classes = torch.where(prediction > 0)[0]
            # if len(instance_classes) == 0: # Nếu instance không thuộc lớp nào, không thêm
            #     return

            # # Tìm lớp đang có số đếm cao nhất trong memory
            # current_counts = self.class_counts[instance_classes]
            # if len(current_counts) == 0:
            #      majority_class_idx = torch.argmax(self.class_counts)
            # else:
            #      majority_class_idx = instance_classes[torch.argmax(current_counts)]

            # # Tìm item tệ nhất trong memory thuộc lớp chiếm ưu thế này để thay thế
            # max_score = -1
            # replace_idx = -1
            # for i, item in enumerate(self.memory):
            #     # Nếu item này thuộc lớp chiếm ưu thế
            #     if item.label[majority_class_idx] > 0:
            #         score = self.heuristic_score(item.age, item.uncertainty)
            #         if score > max_score:
            #             max_score = score
            #             replace_idx = i
            
            # # Nếu không tìm thấy ai để thay thế (trường hợp hiếm), thay thế item có score tệ nhất
            # if replace_idx == -1:
            #     max_score = -1
            #     for i, item in enumerate(self.memory):
            #         score = self.heuristic_score(item.age, item.uncertainty)
            #         if score > max_score:
            #             max_score = score
            #             replace_idx = i

            # # Thực hiện thay thế
            # if replace_idx != -1:
            #     removed_item = self.memory.pop(replace_idx)
            #     # Cập nhật số đếm lớp
            #     self.class_counts -= removed_item.label.long()
                
            #     self.memory.append(new_item)
            #     self.class_counts += new_item.label.long()

            ####### Phiên bản gọn hơn là luôn thay thể cho item có score tệ nhất #######
            # # 1. Tìm item có score tệ nhất (cao nhất) trong TOÀN BỘ memory bank
            # max_score = -1.0
            # replace_idx = -1
            # for i, item in enumerate(self.memory):
            #     score = self.heuristic_score(item.age, item.uncertainty)
            #     if score > max_score:
            #         max_score = score
            #         replace_idx = i
            
            # # 2. Luôn luôn thực hiện thay thế tại vị trí đã tìm thấy
            # if replace_idx != -1:
            #     # Lấy ra item cũ để cập nhật class_counts
            #     removed_item = self.memory.pop(replace_idx)
            #     self.class_counts -= removed_item.label.long()
                
            #     # Thêm item mới vào
            #     self.memory.append(new_item)
            #     self.class_counts += new_item.label.long()

            ####### Phiên bản giữ logic cân bằng lớp #######
            # 1. Tìm lớp chiếm ưu thế nhất (có số lượng item nhiều nhất trong bank)
            majority_class_idx = torch.argmax(self.class_counts).item()

            # 2. Tìm item tệ nhất thuộc lớp chiếm ưu thế đó
            max_score_in_majority = -1.0
            replace_idx = -1
            for i, item in enumerate(self.memory):
                if item.label[majority_class_idx] > 0: # Nếu item thuộc lớp chiếm ưu thế
                    score = self.heuristic_score(item.age, item.uncertainty)
                    if score > max_score_in_majority:
                        max_score_in_majority = score
                        replace_idx = i

            # 3. Nếu không tìm được ai trong lớp chiếm ưu thế (trường hợp hiếm),
            # thì tìm item tệ nhất trong toàn bộ bank.
            if replace_idx == -1:
                max_score_global = -1.0
                for i, item in enumerate(self.memory):
                    score = self.heuristic_score(item.age, item.uncertainty)
                    if score > max_score_global:
                        max_score_global = score
                        replace_idx = i
            
            # 4. Thực hiện thay thế
            if replace_idx != -1:
                removed_item = self.memory.pop(replace_idx)
                self.class_counts -= removed_item.label.long()
                self.memory.append(new_item)
                self.class_counts += new_item.label.long()


    def heuristic_score(self, age, uncertainty):
        # Heuristic score để tìm item TỆ NHẤT (cũ nhất và không chắc chắn nhất)
        # Vì vậy ta dùng age và uncertainty trực tiếp thay vì 1/(...)
        # Lưu ý: uncertainty gốc là entropy (càng cao càng không chắc chắn)
        # age càng cao càng cũ
        # Chúng ta muốn tìm item có age LỚN và uncertainty LỚN
        
        # Chuẩn hóa age để có cùng thang đo với uncertainty
        normalized_age = age / self.capacity
        
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
    
    # Cung cấp phân phối lớp từ `self.class_counts`
    def per_class_dist(self) -> list[int]:
        # Chuyển tensor về list trên CPU để xử lý bên ngoài
        self.class_counts = self._recalculate_class_counts()
        return self.class_counts.cpu().tolist()

    def get_memory(self):
        tmp_data = []
        tmp_age = []

        for item in self.memory:
            tmp_data.append(item.data)
            tmp_age.append(item.age)
            
        # Chuẩn hóa age để dùng trong timeliness_reweighting
        # Chú ý: trong code gốc, age được chia cho capacity ở đây
        # nhưng timeliness_reweighting lại không dùng đến capacity.
        # Để nhất quán, ta truyền age chưa chuẩn hóa
        return tmp_data, tmp_age
    
    def add_instance2(self, instance):
        """
        Thêm một mẫu mới vào memory bank đa nhãn.
        instance: tuple (data, prediction_vector, uncertainty_score)
        """
        assert len(instance) == 3
        x, prediction, uncertainty = instance
        
        # Luôn tăng tuổi trước khi quyết định
        self.add_age()

        # Tạo một MemoryItem và tính score cho nó
        new_item = MemoryItem(data=x, label=prediction, uncertainty=uncertainty, age=0)
        new_score = self.heuristic_score(0, uncertainty)

        # Quyết định xem có nên thêm item mới vào không
        if self._should_add(prediction, new_score):
            self.memory.append(new_item)
            self.class_counts += prediction.long() # Cập nhật số đếm lớp

    def _should_add(self, new_prediction, new_score) -> bool:
        """
        Hàm quyết định chính: có nên thêm item mới không?
        Nếu cần, nó sẽ tự động dọn chỗ.
        """
        if new_prediction.sum() == 0:
            return False

        # Trường hợp 1: Memory chưa đầy, luôn thêm
        if self.get_occupancy() < self.capacity:
            return True

        # Trường hợp 2: Memory đã đầy, cần dọn chỗ
        # Tìm lớp nào đang chiếm ưu thế nhất trong bank
        
        newClassCount = self._recalculate_class_counts()
        majority_class_idx = torch.argmax(newClassCount).item()
        # print(f'newClassCount+++: {newClassCount} - {len(self.memory)}')
        # print(f'Prediction+++: {new_prediction}')

        # Tìm ứng cử viên để xóa: item tệ nhất (score cao nhất) thuộc lớp chiếm ưu thế
        max_score = -1.0
        replace_idx = -1
        
        # Quét để tìm ứng cử viên trong lớp chiếm ưu thế
        for i, item in enumerate(self.memory):
            if item.label[majority_class_idx] > 0: # Nếu item thuộc lớp chiếm ưu thế
                score = self.heuristic_score(item.age, item.uncertainty)
                if score > max_score:
                    max_score = score
                    replace_idx = i
        
        # Nếu không tìm được ai trong lớp chiếm ưu thế (hiếm), tìm item tệ nhất trong toàn bộ bank
        if replace_idx > 0 and max_score > new_score:
            self.memory.pop(replace_idx)
            return True # Dọn chỗ thành công, sẵn sàng để thêm
        # if replace_idx == -1:
        else:
            for i, item in enumerate(self.memory):
                score = self.heuristic_score(item.age, item.uncertainty)
                if score > max_score:
                    max_score = score
                    replace_idx = i

        # Nếu không có gì trong bank để xóa (không thể xảy ra nếu bank đầy), không thêm
        if replace_idx == -1:
            print(f'Co truong hop khong thay idx de thay the: {replace_idx}')
            return False

        # **Logic cốt lõi của RoTTA gốc:**
        # Chỉ thay thế nếu item cũ thực sự tệ hơn item mới
        if max_score > new_score:
            removed_item = self.memory.pop(replace_idx)
            # self.class_counts -= removed_item.label.long()
            return True # Dọn chỗ thành công, sẵn sàng để thêm
        else:
            return False # Item mới không đủ tốt, không thêm
        
        ##### Thu truong hop luon luon thay the memory bank
        # self.memory.pop(replace_idx)
        # return True
        
    def _recalculate_class_counts(self) -> torch.Tensor:
        """
        Tính toán lại và trả về số lượng mẫu cho mỗi lớp dựa trên trạng thái hiện tại của memory.
        """
        if not self.memory:
            return torch.zeros(self.num_class, dtype=torch.long, device=self.device)

        # Lấy tất cả các vector label từ memory
        all_labels = [item.label for item in self.memory]
        
        # Stack chúng lại thành một tensor và tính tổng theo cột
        class_counts = torch.stack(all_labels).long().sum(dim=0)
        return class_counts