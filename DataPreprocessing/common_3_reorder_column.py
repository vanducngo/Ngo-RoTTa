import pandas as pd
import os

def preprocess_csv(input_csv_path, output_csv_path):
    """
    Đọc một file CSV của CheXpert, sắp xếp lại các cột theo thứ tự chuẩn,
    và lưu vào một file mới.
    """
    print(f"Đang đọc file CSV gốc từ: {input_csv_path}")

    try:
        # Đọc dữ liệu từ file CSV gốc
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file tại '{input_csv_path}'. Vui lòng kiểm tra lại đường dẫn.")
        return
    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc file CSV: {e}")
        return

    # In ra các cột gốc để kiểm tra
    original_columns = df.columns.tolist()
    print("\nCác cột gốc:")
    print(original_columns)

    # --- ĐỊNH NGHĨA THỨ TỰ CỘT MỚI ---

    # Cột đầu tiên luôn là 'Path' (hoặc 'image_id' tùy file của bạn)
    # Dựa trên ví dụ của bạn, cột đầu tiên là 'Path'
    # Nếu file của bạn dùng 'image_id', hãy đổi 'Path' thành 'image_id'
    id_column = 'image_id' 
    if id_column not in original_columns:
        if 'image_id' in original_columns:
            id_column = 'image_id'
        else:
            print(f"LỖI: Không tìm thấy cột '{id_column}' hoặc 'image_id' trong file CSV.")
            return

    # 5 cột nhãn mục tiêu bạn muốn đặt lên đầu
    target_labels = [
        'Atelectasis',
        'Cardiomegaly',
        'Consolidation',
        'Pleural Effusion',
        'Pneumothorax',
    ]

    # Lấy danh sách các cột nhãn còn lại (loại trừ các cột đã chọn và cột id)
    # và đặt 'No Finding' ở cuối cùng
    remaining_labels = [col for col in original_columns 
                        if col != id_column and col not in target_labels and col != 'No Finding']
    
    # Sắp xếp các cột còn lại theo thứ tự bảng chữ cái để nhất quán
    remaining_labels.sort()
    
    # Ghép tất cả lại để có thứ tự cột cuối cùng
    # [id] + [5 mục tiêu] + [các cột còn lại] + ['No Finding']
    new_column_order = [id_column] + target_labels + remaining_labels

    if 'No Finding' in original_columns:
        new_column_order += ['No Finding']
    
    # Kiểm tra xem tất cả các cột mới có tồn tại trong các cột gốc không
    for col in new_column_order:
        if col not in original_columns:
            print(f"LỖI: Cột '{col}' được định nghĩa trong thứ tự mới nhưng không tồn tại trong file CSV gốc.")
            return

    # Áp dụng thứ tự cột mới cho DataFrame
    df_reordered = df[new_column_order]

    print("\nCác cột sau khi sắp xếp lại:")
    print(df_reordered.columns.tolist())
    
    # Lưu DataFrame đã sắp xếp lại vào file mới
    try:
        df_reordered.to_csv(output_csv_path, index=False)
        print(f"\nThành công! Đã lưu file đã xử lý vào: {output_csv_path}")
    except Exception as e:
        print(f"\nĐã xảy ra lỗi khi lưu file CSV mới: {e}")


def processingSingleFile(input_file):
    # Lấy thư mục và tên file gốc để tạo tên file mới
    input_dir = os.path.dirname(input_file)
    input_filename = os.path.basename(input_file)
    
    # Tạo tên file đầu ra, ví dụ: "train_processed.csv"
    output_filename = input_filename.replace('.csv', '_reordered.csv')
    output_file = os.path.join(input_dir, output_filename)
    
    # Thực hiện tiền xử lý
    preprocess_csv(input_csv_path=input_file, output_csv_path=output_file)


if __name__ == "__main__":
    # --- CẤU HÌNH ĐƯỜNG DẪN ---
    # Đường dẫn đến file CSV gốc của bạn
    # Dựa trên thông tin bạn cung cấp, cột đầu tiên là Path, không phải image_id
    # input_file = '/home/ngo/Working/Data/CheXpert-v1.0-small/train_restructed.csv'
    # processingSingleFile(input_file)

    # input_file = '/home/ngo/Working/Data/CheXpert-v1.0-small/valid_restructed.csv'
    # processingSingleFile(input_file)

    # input_file = '/home/ngo/Working/Data/MixData/nih_14_structured/validate.csv'
    # processingSingleFile(input_file)

    # input_file = '/home/ngo/Working/Data/MixData/vinbigdata_structured/validate.csv'
    # processingSingleFile(input_file)

    # input_file = '/home/ngoto/Working/Data/CheXpert-v1.0-small/train_final.csv'
    # processingSingleFile(input_file)

    # input_file = '/home/ngoto/Working/Data/CheXpert-v1.0-small/valid_final.csv'
    # processingSingleFile(input_file)

    input_file = '/home/ngoto/Working/Data/MixData/PadChestPruning/refined_padchest_labels.csv'
    processingSingleFile(input_file)