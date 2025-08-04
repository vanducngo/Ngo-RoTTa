import torch
from sklearn.metrics import roc_auc_score
import numpy as np

class AUCProcessor:
    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes

        if class_names is not None and len(class_names) == num_classes:
            self.class_names = class_names
        else:
            if class_names is not None:
                print(f"Warning: `class_names` length ({len(class_names)}) does not match `num_classes` ({num_classes}). Using default names.")
            self.class_names = [f"Class {i}" for i in range(num_classes)]

        self.all_predictions = []
        self.all_labels = []
        self.all_domains = []
        self._calculated = False

    def process(self, predictions, labels, domains):
        self._calculated = False
        
        # self.all_predictions.append(predictions.cpu().numpy())
        # self.all_labels.append(labels.cpu().numpy())
        self.all_predictions.append(predictions.detach().cpu().numpy())
        self.all_labels.append(labels.detach().cpu().numpy())
        
        if isinstance(domains, torch.Tensor):
            domains = [str(d.item()) for d in domains]
        self.all_domains.extend(domains)

    def calculate(self):
        """
        Calculate AUC score after collect all data
        """
        if self._calculated:
            return self.results

        if not self.all_predictions:
            empty_auc_dict = {name: 0.0 for name in self.class_names}
            self.results = {"mean_auc": 0.0, "per_class_auc": empty_auc_dict}
            return self.results

        if isinstance(self.all_predictions, list):
            self.all_predictions = np.concatenate(self.all_predictions, axis=0)
            self.all_labels = np.concatenate(self.all_labels, axis=0)
        
        self.all_labels = self.all_labels.astype(int)

        per_class_auc_dict = {}
        valid_aucs_for_mean = []
        
        for i, class_name in enumerate(self.class_names):
            y_true = self.all_labels[:, i]
            y_pred = self.all_predictions[:, i]
            
            # Check if class ahve both label 0 and 1
            if len(np.unique(y_true)) > 1:
                try:
                    auc = roc_auc_score(y_true, y_pred)
                    per_class_auc_dict[class_name] = auc
                    valid_aucs_for_mean.append(auc)
                except ValueError:
                    per_class_auc_dict[class_name] = float('nan')
            else:
                per_class_auc_dict[class_name] = float('nan')
        
        mean_auc = np.nanmean(valid_aucs_for_mean) if valid_aucs_for_mean else 0.0

        self.results = {
            "mean_auc": mean_auc,
            "per_class_auc": per_class_auc_dict # Lưu lại dictionary
        }
        
        self._calculated = True
        return self.results

    def info(self) -> str:
        self.calculate()        
        info_str = f"Mean AUC: {self.results['mean_auc']:.4f}\n"
        info_str += "Per-class AUC:\n"
        
        per_class_results = self.results.get("per_class_auc", {})
        if per_class_results:
            max_len = max(len(name) for name in per_class_results.keys())
            
            for class_name, auc in per_class_results.items():
                info_str += f"  - {class_name:<{max_len}} : {auc:.4f}\n"
        else:
            info_str += "  No per-class results available.\n"
            
        return info_str.strip()