import pandas as pd
import matplotlib.pyplot as plt
import os

# Định nghĩa đường dẫn đến file CSV đã tinh chỉnh của CheXpert
TRAIN_CSV_PATH = os.path.join("/Users/admin/Working/Data/chexpert_pruning_U_ON", "train.csv")
TEST_CSV_PATH = os.path.join("/Users/admin/Working/Data/chexpert_pruning_U_ON", "valid.csv") 

# Đọc dữ liệu
df_train = pd.read_csv(TRAIN_CSV_PATH)
df_test = pd.read_csv(TEST_CSV_PATH)
print(f"Total rows loaded - Train: {len(df_train)}, Test: {len(df_test)}")

# Định nghĩa các cột nhãn
condition_columns = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Pleural Effusion', 'Pneumothorax']

# Kiểm tra các cột có trong DataFrame
missing_columns_train = [col for col in condition_columns if col not in df_train.columns]
missing_columns_test = [col for col in condition_columns if col not in df_test.columns]
if missing_columns_train or missing_columns_test:
    print(f"Warning: Missing columns - Train: {missing_columns_train}, Test: {missing_columns_test}")
    condition_columns = [col for col in condition_columns if col in df_train.columns and col in df_test.columns]

# Tính tổng số lượng nhãn positive (1.0) và uncertain (0.5) cho mỗi tập
label_distribution_train = df_train[condition_columns].apply(lambda x: x.value_counts()).fillna(0).T
label_distribution_test = df_test[condition_columns].apply(lambda x: x.value_counts()).fillna(0).T

# Tính tổng số mẫu
total_samples_train = len(df_train)
total_samples_test = len(df_test)

# Tính phần trăm positive và uncertain cho train
percent_positive_train = (label_distribution_train.get(1.0, 0) / total_samples_train) * 100
percent_uncertain_train = (label_distribution_train.get(0.5, 0) / total_samples_train) * 100

# Tính phần trăm positive và uncertain cho test
percent_positive_test = (label_distribution_test.get(1.0, 0) / total_samples_test) * 100
percent_uncertain_test = (label_distribution_test.get(0.5, 0) / total_samples_test) * 100

# Tạo DataFrame để so sánh
comparison_df = pd.DataFrame({
    'Train Positive (1.0)': percent_positive_train,
    'Train Uncertain (0.5)': percent_uncertain_train,
    'Test Positive (1.0)': percent_positive_test,
    'Test Uncertain (0.5)': percent_uncertain_test
})

# Vẽ biểu đồ cột trong cùng một hình
plt.figure(figsize=(12, 6))
bar_width = 0.2
index = range(len(condition_columns))

plt.bar([i - bar_width for i in index], comparison_df['Train Positive (1.0)'], bar_width, label='Train Positive (1.0)', color='skyblue')
plt.bar(index, comparison_df['Train Uncertain (0.5)'], bar_width, label='Train Uncertain (0.5)', color='lightgreen')
plt.bar([i + bar_width for i in index], comparison_df['Test Positive (1.0)'], bar_width, label='Test Positive (1.0)', color='lightcoral')
plt.bar([i + 2 * bar_width for i in index], comparison_df['Test Uncertain (0.5)'], bar_width, label='Test Uncertain (0.5)', color='lightyellow')

plt.title('Comparison of Positive (1.0) and Uncertain (0.5) Labels in CheXpert Dataset')
plt.xlabel('Conditions')
plt.ylabel('Percentage (%)')
plt.xticks(index, condition_columns, rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

# Thêm giá trị phần trăm lên cột
for i, col in enumerate(['Train Positive (1.0)', 'Train Uncertain (0.5)', 'Test Positive (1.0)', 'Test Uncertain (0.5)']):
    for j, v in enumerate(comparison_df[col]):
        plt.text(j - bar_width + i * bar_width, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# In kết quả
print("\nLabel Distribution (Percentage):")
print(comparison_df.round(2))