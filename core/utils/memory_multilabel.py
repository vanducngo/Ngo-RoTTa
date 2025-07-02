import torch
import random

class MultiLabelMemory:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = [] # Sẽ chứa các tuple: (image_tensor, pseudo_label_tensor, uncertainty_score, age)

    def __len__(self):
        return len(self.memory)

    def add_instance(self, instance):
        """
        Thêm một mẫu mới vào memory bank.
        instance: tuple (image, pseudo_label, uncertainty)
        """
        # Tăng tuổi cho tất cả các mẫu hiện có
        for i in range(len(self.memory)):
            self.memory[i] = self.memory[i][:3] + (self.memory[i][3] + 1,)

        image, pseudo_label, uncertainty = instance
        age = 0
        
        # Nếu memory chưa đầy, chỉ cần thêm vào
        if len(self.memory) < self.capacity:
            self.memory.append((image, pseudo_label, uncertainty, age))
        else:
            # Nếu memory đã đầy, tìm mẫu "tệ" nhất để thay thế
            # Điểm heuristic: H = uncertainty + weight * age
            # Ở đây ta đơn giản hóa: ưu tiên loại bỏ mẫu có uncertainty cao nhất
            worst_idx = -1
            max_uncertainty = -1
            
            for i, (_, _, u, _) in enumerate(self.memory):
                if u > max_uncertainty:
                    max_uncertainty = u
                    worst_idx = i
            
            # Thay thế mẫu tệ nhất bằng mẫu mới
            self.memory[worst_idx] = (image, pseudo_label, uncertainty, age)
            
    def get_memory_batch(self):
        """Lấy một batch ngẫu nhiên từ memory."""
        if len(self.memory) < self.batch_size:
            return [], [], [] # Trả về rỗng nếu không đủ mẫu

        # Lấy một batch ngẫu nhiên
        batch_samples = random.sample(self.memory, self.batch_size)
        
        images = torch.stack([s[0] for s in batch_samples])
        labels = torch.stack([s[1] for s in batch_samples])
        ages = [s[3] for s in batch_samples]
        
        return images, labels, ages

    def get_occupancy(self):
        return len(self.memory)