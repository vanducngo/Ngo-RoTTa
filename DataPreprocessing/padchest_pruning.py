import pandas as pd
import os
import ast

def refine_padchest_labels(PADCHEST_ROOT_PATH):
  """
  Đọc file CSV gốc của PadChest, lọc và chuyển đổi các nhãn bệnh,
  sau đó lưu vào một file CSV mới đã được tinh chỉnh.

  Hàm này sẽ:
  1. Đọc file 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv'.
  2. Chỉ giữ lại các hàng có chứa ít nhất một trong 5 bệnh được chỉ định.
  3. Tạo ra một file 'refined_padchest_labels.csv' mới với cấu trúc:
     image_id, Atelectasis, Cardiomegaly, Consolidation, Pleural Effusion, Pneumothorax
     với giá trị 1 nếu có bệnh và 0 nếu không.

  Args:
    PADCHEST_ROOT_PATH (str): Đường dẫn đến thư mục chứa file CSV nguồn.
  """
  # --- Cấu hình ---
  source_filename = 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv'
  target_filename = 'refined_padchest_labels.csv'

  # Danh sách các bệnh cần lọc (chữ thường)
  DISEASE_LABELS = [
      'atelectasis',
      'cardiomegaly',
      'consolidation',
      'pleural effusion',
      'pneumothorax'
  ]

  # Tên các cột cuối cùng tương ứng (viết hoa chữ cái đầu)
  FINAL_LABEL_SET_MAPPED = [
      'Atelectasis',
      'Cardiomegaly',
      'Consolidation',
      'Pleural Effusion',
      'Pneumothorax'
  ]

  # Tạo đường dẫn đầy đủ đến file nguồn và file đích
  source_csv_path = os.path.join(PADCHEST_ROOT_PATH, source_filename)
  target_csv_path = os.path.join(PADCHEST_ROOT_PATH, target_filename)

  # --- Bắt đầu xử lý ---
  print(f"Đang đọc file: {source_csv_path}...")
  try:
    # Chỉ đọc các cột cần thiết để tiết kiệm bộ nhớ
    df = pd.read_csv(source_csv_path, usecols=['ImageID', 'Labels'])
  except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file tại '{source_csv_path}'. Vui lòng kiểm tra lại đường dẫn.")
    return
  except ValueError:
    print(f"LỖI: File CSV không chứa các cột 'ImageID' hoặc 'Labels'.")
    return

  # Bước 1: Chuyển đổi cột 'Labels' từ chuỗi thành danh sách Python
  # Xử lý các giá trị NaN (nếu có) bằng cách thay thế chúng bằng một chuỗi danh sách rỗng
  df['Labels'] = df['Labels'].fillna('[]')
  # Áp dụng ast.literal_eval để chuyển đổi chuỗi một cách an toàn
  df['Labels'] = df['Labels'].apply(ast.literal_eval)

  # Bước 2: Tạo các cột mới cho từng bệnh (One-Hot Encoding)
  for i, disease in enumerate(DISEASE_LABELS):
    column_name = FINAL_LABEL_SET_MAPPED[i]
    df[column_name] = df['Labels'].apply(lambda label_list: 1 if disease in label_list else 0)

  # Bước 3: Xóa các hàng không có nhãn nào trong danh sách bệnh
  # Tính tổng các cột bệnh mới. Nếu tổng bằng 0, nghĩa là không có bệnh nào trong danh sách.
  rows_to_keep_mask = df[FINAL_LABEL_SET_MAPPED].sum(axis=1) > 0
  df_filtered = df[rows_to_keep_mask].copy() # Sử dụng .copy() để tránh cảnh báo

  # Bước 4: Tạo DataFrame cuối cùng với các cột và tên cột mong muốn
  final_df = df_filtered.rename(columns={'ImageID': 'image_id'})
  final_df = final_df[['image_id'] + FINAL_LABEL_SET_MAPPED]

  # Bước 5: Ghi DataFrame cuối cùng ra file CSV đích
  final_df.to_csv(target_csv_path, index=False)

  print("-" * 50)
  print("Hoàn tất xử lý!")
  print(f"Số hàng trong file gốc: {len(df)}")
  print(f"Số hàng trong file đã lọc (có ít nhất 1 bệnh): {len(final_df)}")
  print(f"File kết quả đã được lưu tại: {target_csv_path}")
  print("-" * 50)


# --- CÁCH SỬ DỤNG ---

# 1. Thay đổi đường dẫn này thành đường dẫn thực tế trên máy của bạn
#    Ví dụ: PADCHEST_ROOT_PATH = 'D:/datasets/PadChest'
#    hoặc PADCHEST_ROOT_PATH = '/home/user/data/padchest'
PADCHEST_ROOT_PATH = '/home/ngoto/Working/Data/MixData/PadChestPruning' # Sử dụng '.' nếu file CSV nằm cùng thư mục với file Python

# 2. Gọi hàm để thực thi
refine_padchest_labels(PADCHEST_ROOT_PATH)

# (Tùy chọn) In ra 5 dòng đầu của file kết quả để kiểm tra
try:
    result_df = pd.read_csv(os.path.join(PADCHEST_ROOT_PATH, 'refined_padchest_labels.csv'))
    print("\n5 dòng đầu của file kết quả 'refined_padchest_labels.csv':")
    print(result_df.head())
except FileNotFoundError:
    pass