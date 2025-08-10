import pandas as pd
import os
import numpy as np
from skmultilearn.model_selection import IterativeStratification

def create_stratified_subset(input_csv_path, subset_fraction, label_columns, random_seed):
    """
    Hàm chính để tạo một tập con phân tầng.
    Giờ đây nhận `random_seed` làm tham số.
    """
    
    # --- 1. Tạo tên file đầu ra dựa trên seed ---
    input_dir = os.path.dirname(input_csv_path)
    input_filename = os.path.basename(input_csv_path)
    subset_percent = int(subset_fraction * 100)
    # Thêm seed vào tên file để phân biệt các lần chạy
    output_filename = input_filename.replace('.csv', f'_subset_{subset_percent}_seed{random_seed}.csv')
    output_csv_path = os.path.join(input_dir, output_filename)

    # --- 2. Đọc dữ liệu ---
    try:
        print(f"\n--- Chạy với Seed: {random_seed} ---")
        print(f"Đang đọc file dữ liệu từ: {input_csv_path}")
        full_df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file tại '{input_csv_path}'.")
        return
    print(f"Đã đọc thành công {len(full_df)} mẫu.")
    
    # --- 3. Thực hiện Iterative Stratification ---
    print(f"Bắt đầu tạo tập con với tỷ lệ {subset_fraction * 100:.0f}%...")
    
    X = full_df.index.to_numpy().reshape(-1, 1) 
    y = full_df[label_columns].to_numpy()

    num_splits = int(round(1 / subset_fraction))
    
    # Bỏ random_state khỏi hàm khởi tạo để tránh lỗi
    stratifier = IterativeStratification(n_splits=num_splits, order=1)
    
    # Đặt seed cho numpy ngay trước khi chia để kiểm soát tính ngẫu nhiên
    np.random.seed(random_seed)

    try:
        _, subset_indices = next(stratifier.split(X, y))
    except ValueError as e:
        print(f"LỖI khi thực hiện Iterative Stratification với seed {random_seed}: {e}")
        return

    subset_df = full_df.loc[subset_indices].sort_index()

    # --- 4. Lưu và phân tích kết quả ---
    subset_df.to_csv(output_csv_path, index=False)
    print(f"Thành công! Đã lưu tập con với {len(subset_df)} mẫu vào:")
    print(output_csv_path)
    
    # Phân tích sự phân phối để so sánh
    print("\nPhân phối lớp trong tập con mới (%):")
    print((subset_df[label_columns].mean() * 100).round(2))
    print("-" * 30)


if __name__ == "__main__":
    # ==========================================================================
    # --- CẤU HÌNH ---
    
    INPUT_CSV_PATH = "/home/ngoto/Working/Data/MixData/nih_14_structured/validate.csv"
    SUBSET_FRACTION = 0.5
    LABEL_COLUMNS = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 
        'Pleural Effusion', 'Pneumothorax'
    ]
    
    # --- THỬ NGHIỆM VỚI NHIỀU SEED ---
    # Bạn có thể định nghĩa một danh sách các seed muốn thử
    SEEDS_TO_RUN = [42, 123, 2024]
    
    for seed in SEEDS_TO_RUN:
        create_stratified_subset(
            input_csv_path=INPUT_CSV_PATH,
            subset_fraction=SUBSET_FRACTION,
            label_columns=LABEL_COLUMNS,
            random_seed=seed
        )
    # ==========================================================================