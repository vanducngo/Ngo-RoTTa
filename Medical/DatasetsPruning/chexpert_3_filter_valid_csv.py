import pandas as pd

# Đường dẫn file CSV
valid_file = '/Users/admin/Working/Data/CheXpert-v1.0-small/valid.csv'
output_file = '/Users/admin/Working/Data/CheXpert-v1.0-small/valid_filtered.csv'

# Danh sách các bệnh cần kiểm tra
selected_diseases = [
    'No Finding', 
    'Cardiomegaly', 
    'Consolidation', 
    'Pleural Effusion', 
    'Pneumothorax', 
    'Atelectasis'
]

# Đọc file CSV
df = pd.read_csv(valid_file)

# Lọc các dòng mà ít nhất một bệnh trong danh sách có giá trị khác 0.0
df_filtered = df[df[selected_diseases].eq(1.0).any(axis=1)]

# Lưu vào file CSV mới, giữ nguyên tất cả các cột
df_filtered.to_csv(output_file, index=False)

print(f"File đã được lọc và lưu vào {output_file}")