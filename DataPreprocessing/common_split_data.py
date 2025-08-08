import pandas as pd
import os
from sklearn.model_selection import train_test_split

def analyze_label_distribution(df, label_columns, dataset_name="Dataset"):
    """
    Phân tích và in ra số lượng mẫu dương tính, âm tính, và tổng số
    cho từng nhãn trong một DataFrame.

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu.
        label_columns (list): Danh sách các cột nhãn cần phân tích.
        dataset_name (str): Tên của bộ dữ liệu để in ra tiêu đề.
    """
    print(f"\n--- Phân tích phân phối nhãn cho: {dataset_name} ({len(df)} mẫu) ---")
    
    # Tạo một DataFrame để lưu kết quả
    dist_data = []
    for col in label_columns:
        if col in df.columns:
            positive_count = df[col].sum()
            negative_count = len(df) - positive_count
            dist_data.append({
                "Label": col,
                "Positive (1s)": int(positive_count),
                "Negative (0s)": int(negative_count),
                "Total": len(df),
                "Positive (%)": f"{(positive_count / len(df) * 100):.2f}%"
            })
    
    dist_df = pd.DataFrame(dist_data)
    print(dist_df.to_string(index=False)) # .to_string() để in ra đẹp hơn

def create_nih_stratified_subset():
    INPUT_CSV_PATH = "/home/ngoto/Working/Data/MixData/nih_14_structured/validate.csv"

    # 2. Tỷ lệ của tập con (0.2 tương đương 20%)
    SUBSET_FRACTION = 0.2

    # 3. Seed ngẫu nhiên để đảm bảo kết quả luôn giống nhau mỗi khi chạy
    RANDOM_SEED = 42

    # 4. Danh sách các cột nhãn sẽ được dùng để phân tầng.
    #    Đây phải là các cột nhãn có trong file CSV.
    LABEL_COLUMNS = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Pleural Effusion', 'Pneumothorax'
    ]
    # ==========================================================================

    # Tạo tên file đầu ra tự động
    input_dir = os.path.dirname(INPUT_CSV_PATH)
    input_filename = os.path.basename(INPUT_CSV_PATH)
    subset_percent = int(SUBSET_FRACTION * 100)
    output_filename = input_filename.replace('.csv', f'_subset.csv')
    output_csv_path = os.path.join(input_dir, output_filename)

    # --- 1. Đọc dữ liệu ---
    try:
        print(f"Đang đọc file dữ liệu từ: {INPUT_CSV_PATH}")
        full_df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file tại '{INPUT_CSV_PATH}'. Vui lòng kiểm tra lại đường dẫn.")
        return
    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc file CSV: {e}")
        return

    print(f"Đã đọc thành công {len(full_df)} mẫu.")
    
    analyze_label_distribution(full_df, LABEL_COLUMNS, "Dataset")

    # --- 2. Kiểm tra các cột nhãn ---
    for col in LABEL_COLUMNS:
        if col not in full_df.columns:
            print(f"LỖI: Cột nhãn '{col}' không tồn tại trong file CSV.")
            print(f"Các cột có sẵn: {full_df.columns.tolist()}")
            return
            
    # --- 3. Thực hiện lấy mẫu phân tầng ---
    print(f"\nBắt đầu tạo tập con với tỷ lệ {SUBSET_FRACTION * 100:.0f}%...")
    print(f"Phân tầng dựa trên {len(LABEL_COLUMNS)} cột nhãn.")
    print(f"Sử dụng Random Seed: {RANDOM_SEED}")

    X = full_df.index
    y = full_df[LABEL_COLUMNS]

    try:
        _, subset_indices = train_test_split(
            X,
            test_size=SUBSET_FRACTION,
            stratify=y,
            random_state=RANDOM_SEED
        )
    except ValueError as e:
        print("\nLỖI khi thực hiện phân tầng. Điều này thường xảy ra nếu một 'tầng' (stratum)")
        print("(một sự kết hợp duy nhất của các nhãn) chỉ có một mẫu duy nhất, không thể chia được.")
        print(f"Chi tiết lỗi: {e}")
        return
        
    # Lấy ra các hàng tương ứng với các chỉ số đã được chọn và sắp xếp lại
    subset_df = full_df.loc[subset_indices].sort_index()

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
    create_nih_stratified_subset()