import pandas as pd
import os
import numpy as np

# Import công cụ IterativeStratification
try:
    from skmultilearn.model_selection import IterativeStratification
except ImportError:
    print("Lỗi: Thư viện scikit-multilearn chưa được cài đặt.")
    print("Vui lòng chạy: pip install scikit-multilearn")
    exit()

def create_stratified_subset_iterative():
    """
    Hàm chính để đọc file CSV, lấy ra một tập con 20% sử dụng
    Iterative Stratification để duy trì phân phối của 5 bệnh chính.
    """
    
    # ==========================================================================
    # --- CẤU HÌNH CỐ ĐỊNH ---
    
    # 1. Đường dẫn đến file CSV gốc
    INPUT_CSV_PATH = "/home/ngoto/Working/Data/MixData/nih_14_structured/validate.csv"

    # 2. Tỷ lệ của tập con (0.2 tương đương 20%)
    SUBSET_FRACTION = 0.2

    # 3. Seed ngẫu nhiên để đảm bảo kết quả luôn giống nhau mỗi khi chạy
    RANDOM_SEED = 42

    # 4. Danh sách 5 cột nhãn chính sẽ được dùng để phân tầng.
    LABEL_COLUMNS = [
        'Atelectasis',
        'Cardiomegaly',
        'Consolidation',
        'Pleural Effusion',
        'Pneumothorax',
    ]
    # ==========================================================================

    # --- Tự động tạo tên file đầu ra ---
    input_dir = os.path.dirname(INPUT_CSV_PATH)
    input_filename = os.path.basename(INPUT_CSV_PATH)
    subset_percent = int(SUBSET_FRACTION * 100)
    # Thêm hậu tố "_iterative" để phân biệt với phương pháp cũ
    output_filename = input_filename.replace('.csv', f'_subset.csv')
    output_csv_path = os.path.join(input_dir, output_filename)

    # --- 1. Đọc dữ liệu ---
    try:
        print(f"Đang đọc file dữ liệu từ: {INPUT_CSV_PATH}")
        full_df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file tại '{INPUT_CSV_PATH}'. Vui lòng kiểm tra lại đường dẫn.")
        return
    print(f"Đã đọc thành công {len(full_df)} mẫu.")
    
    # --- 2. Kiểm tra các cột nhãn ---
    for col in LABEL_COLUMNS:
        if col not in full_df.columns:
            print(f"LỖI: Cột nhãn '{col}' không tồn tại trong file CSV.")
            return
            
    # --- 3. Thực hiện Iterative Stratification ---
    print(f"\nBắt đầu tạo tập con với tỷ lệ {SUBSET_FRACTION * 100:.0f}%...")
    print(f"Sử dụng Iterative Stratification trên {len(LABEL_COLUMNS)} cột nhãn.")
    print(f"Sử dụng Random Seed: {RANDOM_SEED}")

    # Chuẩn bị dữ liệu cho stratifier
    # X có thể là bất cứ thứ gì, chỉ cần có cùng số dòng. Ta dùng index.
    X = full_df.index.to_numpy().reshape(-1, 1) 
    # y phải là một mảng numpy
    y = full_df[LABEL_COLUMNS].to_numpy()

    # Khởi tạo bộ chia. n_splits sẽ xác định tỷ lệ.
    # Để lấy 20%, ta chia thành 5 phần (1/0.2 = 5)
    num_splits = int(round(1 / SUBSET_FRACTION))
    
    stratifier = IterativeStratification(n_splits=num_splits, order=1, random_state=RANDOM_SEED)

    try:
        # Lấy chỉ số của phần còn lại (rest) và phần tập con (subset)
        # stratifier.split trả về một generator, ta chỉ cần lấy lần chia đầu tiên
        rest_indices, subset_indices = next(stratifier.split(X, y))
    except ValueError as e:
        print(f"\nLỖI khi thực hiện Iterative Stratification: {e}")
        print("Điều này vẫn có thể xảy ra nếu dữ liệu quá thưa thớt hoặc có vấn đề.")
        return

    # Lấy ra các hàng tương ứng với các chỉ số đã được chọn
    subset_df = full_df.iloc[subset_indices].sort_index()

    # --- 4. Lưu và phân tích kết quả ---
    try:
        subset_df.to_csv(output_csv_path, index=False)
        print(f"\nThành công! Đã tạo và lưu tập con với {len(subset_df)} mẫu vào:")
        print(output_csv_path)
    except Exception as e:
        print(f"\nĐã xảy ra lỗi khi lưu file CSV mới: {e}")
        return

    # Phân tích sự phân phối để so sánh
    print("\n--- Phân tích phân phối ---")
    comparison_df = pd.DataFrame({
        f'Original ({len(full_df)} samples) %': full_df[LABEL_COLUMNS].mean() * 100,
        f'Subset ({len(subset_df)} samples) %': subset_df[LABEL_COLUMNS].mean() * 100
    })
    comparison_df['Difference'] = comparison_df['Subset (%)'] - comparison_df['Original (%)']
    print(comparison_df.round(2))
    print("\nCột 'Difference' cho thấy sự khác biệt về tỷ lệ phần trăm của mỗi lớp.")
    print("Các giá trị gần 0 cho thấy sự phân phối đã được duy trì rất tốt.")


if __name__ == "__main__":
    create_stratified_subset_iterative()