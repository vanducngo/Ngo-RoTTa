import pandas as pd

# ==============================================================================
# BƯỚC 1: ĐỊNH NGHĨA BỘ NHÃN CHUNG
# ==============================================================================
# Chọn 5 bệnh lý phổ biến và nhất quán nhất
COMMON_DISEASES = [
    'No Finding',
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Pleural Effusion',
    'Pneumothorax'
]


# ==============================================================================
# BƯỚC 2: VIẾT HÀM MAPPING CHO TỪNG BỘ DỮ LIỆU
# ==============================================================================

def map_chexpert_labels(df_raw):
    """
    Ánh xạ nhãn cho bộ dữ liệu CheXpert.
    - Xử lý giá trị không chắc chắn (-1).
    - Giữ lại các cột trong COMMON_DISEASES.
    """
    # CheXpert đã có sẵn tất cả các cột cần thiết với tên chuẩn
    # Chỉ cần xử lý các giá trị không chắc chắn (-1.0)
    
    df_mapped = df_raw[['Path'] + COMMON_DISEASES].copy()
    df_mapped = df_mapped.fillna(0) # Coi NaN là 0

    for col in COMMON_DISEASES:
        if col in df_mapped.columns:
            # Chiến lược phổ biến: Coi "không chắc chắn" là "dương tính" để không bỏ lỡ bệnh
            # Hoặc bạn có thể coi là 0: df_mapped[col] = df_mapped[col].replace(-1.0, 0)
            df_mapped[col] = df_mapped[col].replace(-1.0, 1.0)
            
    return df_mapped

def map_vindr_labels(df_raw):
    return df_raw

def map_vindr_labels_bk(df_raw):
    """
    Ánh xạ nhãn cho VinDr-CXR.
    - Chuyển từ định dạng "dài" sang "rộng".
    - Chuẩn hóa tên bệnh.
    - Tạo cột 'No Finding'.
    """
    # Chuẩn hóa tên cột có thể có trong file CSV gốc
    vindr_to_common_map = {
        'Atelectasis': 'Atelectasis',
        'Cardiomegaly': 'Cardiomegaly',
        'Consolidation': 'Consolidation',
        'Pleural effusion': 'Pleural Effusion', # Chú ý khoảng trắng và chữ thường
        'Pneumothorax': 'Pneumothorax'
    }
    df_raw['class_name'] = df_raw['class_name'].replace(vindr_to_common_map)

    # Lọc ra các hàng có bệnh mà chúng ta quan tâm
    df_findings = df_raw[df_raw['class_name'].isin(COMMON_DISEASES)].copy()

    # Pivot để tạo ma trận nhãn đa lớp
    df_pivot = df_findings.pivot_table(
        index='image_id', 
        columns='class_name', 
        aggfunc='size', 
        fill_value=0
    ).reset_index()

    # Thêm lại tất cả các ảnh để xác định 'No Finding'
    all_image_ids = df_raw[['image_id']].drop_duplicates()
    df_wide = pd.merge(all_image_ids, df_pivot, on='image_id', how='left')
    df_wide.fillna(0, inplace=True)

    # Tạo cột 'No Finding'
    # Một ảnh là 'No Finding' nếu tổng các bệnh trong COMMON_DISEASES bằng 0
    df_wide['No Finding'] = (df_wide[COMMON_DISEASES].sum(axis=1) == 0).astype(int)
    
    # Đảm bảo tất cả các cột trong COMMON_DISEASES đều tồn tại
    for label in COMMON_DISEASES:
        if label not in df_wide.columns:
            df_wide[label] = 0

    # Giữ lại các cột cần thiết theo đúng thứ tự
    df_mapped = df_wide[['image_id'] + COMMON_DISEASES].copy()
    
    return df_mapped


def map_chestxray14_labels(df_raw):
    """
    Ánh xạ nhãn cho Chest X-ray14 (NIH).
    - Gộp nhiều nhãn gốc thành một nhãn chung.
    - Chuẩn hóa tên.
    """
    # Đổi tên cột 'Finding Labels' để dễ xử lý
    df_raw.rename(columns={'Finding Labels': 'labels'}, inplace=True)
    
    # Tạo các cột nhị phân cho mỗi bệnh trong bộ nhãn chung
    for disease in COMMON_DISEASES:
        # Xử lý trường hợp đặc biệt của Pleural Effusion
        if disease == 'Pleural Effusion':
            df_raw[disease] = df_raw['labels'].apply(lambda x: 1 if 'Effusion' in x else 0)
        else:
            df_raw[disease] = df_raw['labels'].apply(lambda x: 1 if disease in x else 0)
            
    # Tạo cột 'No Finding'
    df_raw['No Finding'] = df_raw['labels'].apply(lambda x: 1 if 'No Finding' in x else 0)
    
    # Giữ lại các cột cần thiết
    df_mapped = df_raw[['Image Index'] + COMMON_DISEASES].copy()
    # Đổi tên cột 'Image Index' thành 'image_id' cho nhất quán
    df_mapped.rename(columns={'Image Index': 'image_id'}, inplace=True)
    
    return df_mapped


def map_padchest_labels(df_raw):
    """
    Ánh xạ nhãn cho PadChest. Đây là trường hợp phức tạp nhất.
    - Lọc các hàng có nhãn liên quan.
    - Pivot và xử lý 'No Finding'.
    """
    # Chuẩn hóa tên cột nhãn trong PadChest (chữ thường)
    padchest_common_diseases = [d.lower() for d in COMMON_DISEASES]
    
    # Lọc các hàng có nhãn liên quan
    # 'Labels' là cột chứa một list các string
    def filter_labels(label_list):
        if isinstance(label_list, str):
            # Chuyển string dạng list thành list thực sự
            label_list = eval(label_list)
        return any(label in padchest_common_diseases for label in label_list)
    
    df_filtered = df_raw[df_raw['Labels'].apply(filter_labels)].copy()
    
    # Tạo các cột nhị phân cho mỗi bệnh
    for disease_common, disease_padchest in zip(COMMON_DISEASES, padchest_common_diseases):
        def has_disease(label_list):
            if isinstance(label_list, str):
                label_list = eval(label_list)
            return 1 if disease_padchest in label_list else 0
        df_filtered[disease_common] = df_filtered['Labels'].apply(has_disease)
        
    # Xử lý 'No Finding'
    # PadChest có thể có nhãn 'normal' hoặc không có nhãn bệnh lý nào
    def is_no_finding(label_list):
        if isinstance(label_list, str):
            label_list = eval(label_list)
        return 1 if 'normal' in label_list or not any(d in label_list for d in padchest_common_diseases) else 0

    # Chúng ta cần làm việc trên df_raw để xác định No Finding một cách chính xác
    df_raw['No Finding'] = df_raw['Labels'].apply(is_no_finding)
    
    # Gộp thông tin
    df_wide = pd.merge(df_raw[['ImageID', 'No Finding']], 
                       df_filtered[['ImageID'] + COMMON_DISEASES], 
                       on='ImageID', 
                       how='left').fillna(0)
    df_wide = df_wide.drop_duplicates(subset=['ImageID'])
                       
    df_mapped = df_wide[['ImageID'] + COMMON_DISEASES].copy()
    df_mapped.rename(columns={'ImageID': 'image_id'}, inplace=True)

    return df_mapped