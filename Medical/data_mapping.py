import pandas as pd

# ----- ĐỊNH NGHĨA NHÃN CHUNG -----
# Đây là "nguồn chân lý" (source of truth) cho toàn bộ dự án
COMMON_DISEASES = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Pleural Effusion',
    'Pneumothorax',
    'Nodule/Mass'
]

VINDR_CLASSES = [
    'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
    'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
    'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
    'Pulmonary fibrosis'
]

def map_chexpert_labels(df):
    """
    Ánh xạ và chuẩn hóa nhãn cho bộ dữ liệu CheXpert.
    CheXpert đã có hầu hết các nhãn, chỉ cần chọn cột và xử lý "Nodule/Mass".
    """
    # CheXpert có "Lung Lesion" có thể coi là tương đương với "Nodule/Mass"
    df = df.rename(columns={'Lung Lesion': 'Nodule/Mass'})
    
    # Giữ lại các cột cần thiết, bao gồm cả cột Path
    df = df[['Path'] + COMMON_DISEASES]
    
    # Xử lý các giá trị NaN và -1 (uncertain)
    # Cách đơn giản: coi NaN và -1 là 0 (âm tính)
    for col in COMMON_DISEASES:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            df[col] = df[col].replace(-1.0, 0)
        else:
            # Nếu vì lý do nào đó cột không tồn tại, tạo cột mới với giá trị 0
            df[col] = 0
            
    return df[df.columns.intersection(['Path'] + COMMON_DISEASES)]

def map_vindr_labels(df_raw):
    """
    Chuyển đổi DataFrame của VinDr-CXR từ định dạng "dài" sang "rộng" 
    và ánh xạ về các lớp bệnh chung.
    """
    # Bước 1: Chỉ giữ lại các hàng có bệnh mà chúng ta quan tâm
    # Lọc ra các hàng có class_name nằm trong VINDR_CLASSES (loại bỏ 'No finding')
    df_findings = df_raw[df_raw['class_name'].isin(VINDR_CLASSES)].copy()

    # Bước 2: Tạo ma trận nhãn đa lớp (multi-label one-hot encoding)
    # Dùng pivot_table để biến đổi dữ liệu
    # - index: mỗi hàng là một image_id duy nhất
    # - columns: mỗi cột là một class_name
    # - values: đặt là 1 nếu có bệnh, 0 nếu không
    # - aggfunc='size' để đếm, fill_value=0 để điền vào các ô trống
    df_pivot = df_findings.pivot_table(
        index='image_id', 
        columns='class_name', 
        aggfunc='size', 
        fill_value=0
    ).astype(int).reset_index()

    # Bước 3: Xử lý các ảnh không có phát hiện bệnh nào (No finding)
    # Lấy danh sách tất cả các ID ảnh duy nhất từ file gốc
    all_image_ids = df_raw[['image_id']].drop_duplicates()
    
    # Merge df_pivot với tất cả các ID để đảm bảo không bỏ sót ảnh "No finding"
    # Các ảnh không có trong df_pivot sẽ có giá trị NaN ở các cột bệnh
    df_wide = pd.merge(all_image_ids, df_pivot, on='image_id', how='left')
    
    # Điền 0 cho tất cả các giá trị NaN (tức là các ảnh không có bệnh nào trong danh sách)
    df_wide.fillna(0, inplace=True)
    
    # Bước 4: Ánh xạ tên cột bệnh của VinDr về tên chung (nếu cần)
    # Trong trường hợp này, tên cột đã khá chuẩn, nhưng ta vẫn nên có bước này
    # để đảm bảo tính nhất quán.
    vindr_to_common_map = {
        'Cardiomegaly': 'Cardiomegaly',
        'Consolidation': 'Consolidation',
        'Atelectasis': 'Atelectasis',
        'Pleural effusion': 'Pleural Effusion', # Chú ý khoảng trắng
        'Pneumothorax': 'Pneumothorax',
        'Nodule/Mass': 'Nodule/Mass'
    }
    # Đổi tên cột
    df_wide.rename(columns=vindr_to_common_map, inplace=True)

    # Đổi tên cột "Pleural effusion" thành "Pleural Effusion"
    if 'Pleural effusion' in df_wide.columns:
        df_wide.rename(columns={'Pleural effusion': 'Pleural Effusion'}, inplace=True)
        
    # Bước 5: Đảm bảo tất cả các cột trong COMMON_DISEASES đều tồn tại
    for disease in COMMON_DISEASES:
        if disease not in df_wide.columns:
            df_wide[disease] = 0 # Thêm cột bệnh bị thiếu và điền giá trị 0
            
    # Bước 6: Chỉ giữ lại các cột cần thiết: image_id và các bệnh chung
    df_mapped = df_wide[['image_id'] + COMMON_DISEASES].copy()
    
    return df_mapped

def map_chestxray14_labels(df):
    """
    Ánh xạ và chuẩn hóa nhãn cho bộ dữ liệu ChestX-ray14 (NIH).
    """
    # Đổi tên cột 'Finding Labels' để dễ xử lý
    df = df.rename(columns={'Finding Labels': 'Labels', 'Image Index': 'image_id'})
    
    # Tạo các cột riêng cho từng bệnh từ cột 'Labels' (dạng one-hot encoding)
    for disease in COMMON_DISEASES:
        # Xử lý trường hợp tên nhãn khác nhau
        if disease == 'Pleural Effusion':
            # ChestX-ray14 dùng 'Effusion'
            df[disease] = df['Labels'].apply(lambda x: 1 if 'Effusion' in x else 0)
        elif disease == 'Nodule/Mass':
            # Kết hợp 'Nodule' và 'Mass' thành một nhãn
            df[disease] = df['Labels'].apply(lambda x: 1 if ('Nodule' in x or 'Mass' in x) else 0)
        elif disease == 'Pleural Thickening':
             # ChestX-ray14 dùng 'Pleural_Thickening'
             df[disease] = df['Labels'].apply(lambda x: 1 if 'Pleural_Thickening' in x else 0)
        else:
            df[disease] = df['Labels'].apply(lambda x: 1 if disease in x else 0)
            
    # Giữ lại các cột cần thiết
    df = df[['image_id'] + COMMON_DISEASES]
    
    return df