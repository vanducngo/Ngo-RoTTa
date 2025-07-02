import torch
import random
import numpy as np
from collections import defaultdict

class AdvancedMultiLabelMemory:
    def __init__(self, cfg, num_classes, class_names):
        self.capacity = cfg.ADAPTER.MEMORY_SIZE
        self.batch_size = cfg.ADAPTER.BATCH_SIZE
        self.num_classes = num_classes
        self.class_names = class_names
        
        # Ngưỡng để chấp nhận một mẫu vào memory
        self.uncertainty_threshold = cfg.ADAPTER.UNCERTAINTY_THRESHOLD
        
        # Trọng số cho tuổi khi tính điểm heuristic
        self.age_weight = cfg.ADAPTER.AGE_WEIGHT
        
        # Memory được tổ chức theo từng lớp
        # self.memory[class_idx] sẽ là một list các tuple
        # tuple: (image_tensor, full_pseudo_label_tensor, uncertainty_score, age)
        self.memory = defaultdict(list)
        
        # Tính toán sức chứa cho mỗi lớp
        # Chia đều dung lượng cho các lớp bệnh, 'No Finding' có thể có dung lượng riêng
        self.capacity_per_class = self.capacity // self.num_classes
        
        print(f"Initialized Advanced Memory Bank. Capacity: {self.capacity}, Per-Class Capacity: {self.capacity_per_class}")

    def __len__(self):
        # Trả về tổng số mẫu duy nhất trong memory
        all_images = {id(s[0]) for M_c in self.memory.values() for s in M_c}
        return len(all_images)

    def add_instance(self, instance):
        """
        Thêm một mẫu mới vào các "ngăn" tương ứng trong memory bank.
        instance: tuple (image_tensor, full_pseudo_label_tensor, uncertainty_score)
        """
        image, pseudo_label, uncertainty = instance
        
        # === BƯỚC 1: LỌC THEO UNCERTAINTY ===
        # Chỉ xem xét các mẫu có độ chắc chắn đủ cao
        if uncertainty > self.uncertainty_threshold:
            return # Bỏ qua mẫu không đáng tin cậy

        # === BƯỚC 2: TĂNG TUỔI CHO CÁC MẪU HIỆN CÓ ===
        for c in range(self.num_classes):
            for i in range(len(self.memory[c])):
                self.memory[c][i] = self.memory[c][i][:3] + (self.memory[c][i][3] + 1,)

        # === BƯỚC 3: THÊM MẪU MỚI VÀO CÁC NGĂN PHÙ HỢP ===
        age = 0
        new_sample = (image, pseudo_label, uncertainty, age)

        # Lặp qua từng lớp để xem có nên thêm mẫu vào "ngăn" của lớp đó không
        for c_idx in range(self.num_classes):
            # Nếu nhãn giả của lớp này là dương tính
            if pseudo_label[c_idx] == 1:
                class_memory = self.memory[c_idx]
                
                # Nếu ngăn chưa đầy, thêm vào
                if len(class_memory) < self.capacity_per_class:
                    class_memory.append(new_sample)
                else:
                    # Nếu ngăn đã đầy, tìm mẫu tệ nhất để thay thế
                    worst_idx, max_heuristic_score = -1, -1.0
                    
                    for i, (_, _, u, a) in enumerate(class_memory):
                        # Điểm heuristic: H = uncertainty + weight * age
                        # Càng cao càng tệ
                        heuristic_score = u + self.age_weight * a
                        if heuristic_score > max_heuristic_score:
                            max_heuristic_score = heuristic_score
                            worst_idx = i
                            
                    # So sánh điểm của mẫu mới với mẫu tệ nhất
                    new_heuristic_score = uncertainty + self.age_weight * age # age = 0
                    if new_heuristic_score < max_heuristic_score:
                        class_memory[worst_idx] = new_sample

    def get_memory_batch(self):
        """
        Lấy một batch cân bằng từ memory.
        Chiến lược: Lấy một vài mẫu từ mỗi "ngăn" (mỗi lớp).
        """
        if len(self) < self.batch_size:
            return [], [], []

        # Tạo một danh sách phẳng chứa tất cả các mẫu duy nhất
        # Dùng dict để loại bỏ các ảnh trùng lặp
        all_unique_samples = {id(s[0]): s for M_c in self.memory.values() for s in M_c}
        
        if len(all_unique_samples) < self.batch_size:
            return [], [], []

        # Lấy một batch ngẫu nhiên từ các mẫu duy nhất
        batch_samples = random.sample(list(all_unique_samples.values()), self.batch_size)
        
        images = torch.stack([s[0] for s in batch_samples])
        labels = torch.stack([s[1] for s in batch_samples])
        ages = [s[3] for s in batch_samples]
        
        return images, labels, ages

    def get_occupancy(self):
        """Trả về số lượng mẫu trong mỗi ngăn."""
        return {self.class_names[c]: len(mem) for c, mem in self.memory.items()}
    
    def get_list_class_name(self):
        """Trả về số lượng mẫu trong mỗi ngăn."""
        return [name for name in self.class_names]