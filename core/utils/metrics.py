import torch
from sklearn.metrics import roc_auc_score
import numpy as np

class AUCProcessor:
    def __init__(self, num_classes, class_names=None):
        """
        Khởi tạo processor.

        Args:
            num_classes (int): Tổng số lớp.
            class_names (list, optional): Danh sách tên các lớp. 
                                          Thứ tự phải khớp với các cột của tensor.
                                          Mặc định là None.
        """
        self.num_classes = num_classes

        # --- SỬA ĐỔI 1: Lưu lại danh sách tên lớp ---
        # Kiểm tra xem class_names có hợp lệ không. Nếu không, tạo tên mặc định.
        if class_names is not None and len(class_names) == num_classes:
            self.class_names = class_names
        else:
            if class_names is not None:
                print(f"Warning: `class_names` length ({len(class_names)}) does not match `num_classes` ({num_classes}). Using default names.")
            self.class_names = [f"Class {i}" for i in range(num_classes)]
        # --- KẾT THÚC SỬA ĐỔI 1 ---

        self.all_predictions = []
        self.all_labels = []
        self.all_domains = []
        # Thêm một cờ để tránh tính toán lại không cần thiết
        self._calculated = False

    def process(self, predictions, labels, domains):
        """
        Lưu trữ các dự đoán (xác suất) và nhãn thật cho một batch.
        """
        # Reset cờ nếu có dữ liệu mới được thêm vào
        self._calculated = False
        
        self.all_predictions.append(predictions.cpu().numpy())
        self.all_labels.append(labels.cpu().numpy())
        
        # Xử lý domains một cách an toàn
        if isinstance(domains, torch.Tensor):
            domains = [str(d.item()) for d in domains]
        self.all_domains.extend(domains)

    def calculate(self):
        """
        Tính toán AUC score sau khi thu thập tất cả dữ liệu.
        """
        # Nếu đã tính rồi thì không cần tính lại
        if self._calculated:
            return self.results

        if not self.all_predictions:
            # Tạo kết quả rỗng với cấu trúc đúng
            empty_auc_dict = {name: 0.0 for name in self.class_names}
            self.results = {"mean_auc": 0.0, "per_class_auc": empty_auc_dict}
            return self.results

        # Nối tất cả các batch lại thành một mảng lớn
        # Chỉ thực hiện một lần
        if isinstance(self.all_predictions, list):
            self.all_predictions = np.concatenate(self.all_predictions, axis=0)
            self.all_labels = np.concatenate(self.all_labels, axis=0)
        
        self.all_labels = self.all_labels.astype(int)

        # --- SỬA ĐỔI 2: Sử dụng dictionary để lưu kết quả theo tên lớp ---
        per_class_auc_dict = {}
        valid_aucs_for_mean = []
        
        for i, class_name in enumerate(self.class_names):
            y_true = self.all_labels[:, i]
            y_pred = self.all_predictions[:, i]
            
            # Kiểm tra xem lớp này có cả nhãn 0 và 1 không
            if len(np.unique(y_true)) > 1:
                try:
                    auc = roc_auc_score(y_true, y_pred)
                    per_class_auc_dict[class_name] = auc
                    valid_aucs_for_mean.append(auc)
                except ValueError:
                    per_class_auc_dict[class_name] = float('nan')
            else:
                per_class_auc_dict[class_name] = float('nan')
        # --- KẾT THÚC SỬA ĐỔI 2 ---
        
        # Tính mean AUC, bỏ qua các lớp NaN
        mean_auc = np.nanmean(valid_aucs_for_mean) if valid_aucs_for_mean else 0.0

        self.results = {
            "mean_auc": mean_auc,
            "per_class_auc": per_class_auc_dict # Lưu lại dictionary
        }
        
        # Đặt cờ
        self._calculated = True

        return self.results

    def info(self) -> str:
        """
        Trả về một chuỗi có định dạng đẹp để in ra console hoặc ghi vào log.
        """
        # Luôn gọi calculate() để đảm bảo có kết quả mới nhất
        self.calculate()
        
        # --- SỬA ĐỔI 3: Định dạng output dựa trên tên lớp ---
        info_str = f"Mean AUC: {self.results['mean_auc']:.4f}\n"
        info_str += "Per-class AUC:\n"
        
        per_class_results = self.results.get("per_class_auc", {})
        if per_class_results:
            # Tìm độ dài tên lớp dài nhất để căn chỉnh cho đẹp
            max_len = max(len(name) for name in per_class_results.keys())
            
            for class_name, auc in per_class_results.items():
                # Dùng f-string để căn chỉnh cột
                info_str += f"  - {class_name:<{max_len}} : {auc:.4f}\n"
        else:
            info_str += "  No per-class results available.\n"
        # --- KẾT THÚC SỬA ĐỔI 3 ---
            
        return info_str.strip()