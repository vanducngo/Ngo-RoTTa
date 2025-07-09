import torch
from sklearn.metrics import roc_auc_score
import numpy as np

class AUCProcessor:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.all_predictions = []
        self.all_labels = []
        self.all_domains = []

    def process(self, predictions, labels, domains):
        """
        Lưu trữ các dự đoán (xác suất) và nhãn thật.
        predictions: torch.Tensor, shape (batch_size, num_classes), là xác suất sau sigmoid.
        labels: torch.Tensor, shape (batch_size, num_classes), là nhãn one-hot.
        domains: torch.Tensor or list, chứa domain index cho từng mẫu.
        """
        self.all_predictions.append(predictions.cpu().numpy())
        self.all_labels.append(labels.cpu().numpy())
        self.all_domains.extend(domains.cpu().numpy() if isinstance(domains, torch.Tensor) else domains)

    def calculate(self):
        """
        Tính toán AUC score sau khi thu thập tất cả dữ liệu.
        """
        if not self.all_predictions:
            return {"mean_auc": 0.0, "per_class_auc": [0.0] * self.num_classes}

        # Nối tất cả các batch lại thành một mảng lớn
        self.all_predictions = np.concatenate(self.all_predictions, axis=0)
        self.all_labels = np.concatenate(self.all_labels, axis=0)
        
        # Đảm bảo các nhãn là kiểu integer
        self.all_labels = self.all_labels.astype(int)

        per_class_auc = []
        valid_classes = 0
        for i in range(self.num_classes):
            # Kiểm tra xem lớp này có cả nhãn 0 và 1 không
            if len(np.unique(self.all_labels[:, i])) > 1:
                auc = roc_auc_score(self.all_labels[:, i], self.all_predictions[:, i])
                per_class_auc.append(auc)
                valid_classes += 1
            else:
                # Nếu một lớp chỉ có một loại nhãn, AUC không xác định. Gán là NaN hoặc 0.
                per_class_auc.append(float('nan'))

        # Tính mean AUC, bỏ qua các lớp NaN
        mean_auc = np.nanmean(per_class_auc)

        self.results = {
            "mean_auc": mean_auc,
            "per_class_auc": per_class_auc
        }

    def info(self):
        if not hasattr(self, 'results'):
            self.calculate()
        
        info_str = f"Mean AUC: {self.results['mean_auc']:.4f}\n"
        info_str += "Per-class AUC:\n"
        for i, auc in enumerate(self.results['per_class_auc']):
            info_str += f"  Class {i}: {auc:.4f}\n"
        return info_str