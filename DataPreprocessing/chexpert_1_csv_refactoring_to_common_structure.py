import pandas as pd

def process_chexpert_data(input_csv_path, output_csv_path):
    """
    Hàm đọc file CSV của CheXpert, xử lý nhãn và cột, sau đó lưu ra file mới.

    Các bước xử lý:
    1. Đọc file CSV.
    2. Đổi tên cột 'Path' thành 'image_id'.
    3. Xóa các cột metadata: 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA'.
    4. Áp dụng quy tắc gán nhãn "Uncertain as 0.5" cho các cột bệnh:
       - Giá trị rỗng (NaN) -> 0.0
       - Giá trị -1.0 (không chắc chắn) -> 0.5
    5. Lưu DataFrame đã xử lý ra file CSV mới.

    Args:
        input_csv_path (str): Đường dẫn đến file CSV gốc.
        output_csv_path (str): Đường dẫn để lưu file CSV đã xử lý.
    """
    print(f"Bắt đầu xử lý file: {input_csv_path}...")
    
    # Bước 1: Đọc file CSV
    df = pd.read_csv(input_csv_path)

    # Bước 2: Đổi tên cột 'Path' thành 'image_id'
    print("Đang đổi tên cột 'Path' -> 'image_id'...")
    df = df.rename(columns={'Path': 'image_id'})
    
    # Bước 3: Xác định và xóa các cột metadata không cần thiết
    columns_to_drop = ['Sex', 'Age', 'Frontal/Lateral', 'AP/PA']
    print(f"Đang xóa các cột: {columns_to_drop}...")
    df = df.drop(columns=columns_to_drop)

    # Bước 4: Xử lý các cột bệnh
    # Xác định các cột bệnh (tất cả các cột còn lại trừ 'image_id')
    finding_cols = df.columns[1:]
    
    print("Đang xử lý các nhãn bệnh theo quy tắc 'uncertain as 0.5'...")
    for col in finding_cols:
        # Điền các giá trị rỗng (NaN) bằng 0.0
        df[col] = df[col].fillna(0.0)
        
        # Thay thế giá trị -1.0 (không chắc chắn) bằng 0.5
        df[col] = df[col].replace(-1.0, 0.0)

    # Bước 5: Lưu DataFrame đã được xử lý
    df.to_csv(output_csv_path, index=False)
    
    print(f"Xử lý hoàn tất! File đã được lưu tại: {output_csv_path}")
    print("Định dạng file đầu ra:")
    print(df.head()) # In ra 5 dòng đầu tiên để kiểm tra
    print("-" * 50)


# --- CÁCH SỬ DỤNG ---

# Giả sử bạn có file 'train.csv' và 'valid.csv' trong cùng thư mục với file script
# Đường dẫn file đầu vào
train_file = '/home/ngoto/Working/Data/CheXpert-v1.0-small/train.csv'
valid_file = '/home/ngoto/Working/Data/CheXpert-v1.0-small/valid.csv'

# Đường dẫn file đầu ra (tên file mới)
fixed_train_file = '/home/ngoto/Working/Data/CheXpert-v1.0-small/train_restructed.csv'
fixed_valid_file = '/home/ngoto/Working/Data/CheXpert-v1.0-small/valid_restructed.csv'

# Chạy hàm để xử lý file train.csv
process_chexpert_data(train_file, fixed_train_file)

# Chạy hàm để xử lý file valid.csv
process_chexpert_data(valid_file, fixed_valid_file)