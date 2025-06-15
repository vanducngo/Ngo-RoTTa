import pandas as pd

# ==============================================================================
# BƯỚC 1: ĐỊNH NGHĨA BỘ NHÃN CHUNG
# ==============================================================================
# Chọn 5 bệnh lý phổ biến và nhất quán nhất
DISEASES = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Pleural Effusion',
    'Pneumothorax'
]
FINAL_LABEL_SET = DISEASES + ['No Finding']

# ==============================================================================
# BƯỚC 2: VIẾT HÀM MAPPING CHO TỪNG BỘ DỮ LIỆU
# ==============================================================================

def map_chexpert_labels(df_raw):
    # Chỉ cần đảm bảo các cột cuối cùng khớp với FINAL_LABEL_SET
    df_mapped = df_raw[['Path'] + FINAL_LABEL_SET].copy()
    df_mapped = df_mapped.fillna(0)
    for col in FINAL_LABEL_SET:
        if col in df_mapped.columns:
            df_mapped[col] = df_mapped[col].replace(-1.0, 1.0)
    return df_mapped

def map_vindr_labels(df_raw):
    vindr_to_common_map = {
        'Atelectasis': 'Atelectasis', 'Cardiomegaly': 'Cardiomegaly',
        'Consolidation': 'Consolidation', 'Pleural effusion': 'Pleural Effusion',
        'Pneumothorax': 'Pneumothorax'
    }
    df_raw['class_name'] = df_raw['class_name'].replace(vindr_to_common_map)
    df_findings = df_raw[df_raw['class_name'].isin(DISEASES)].copy()
    df_pivot = df_findings.pivot_table(index='image_id', columns='class_name', aggfunc='size', fill_value=0).reset_index()
    all_image_ids = df_raw[['image_id']].drop_duplicates()
    df_wide = pd.merge(all_image_ids, df_pivot, on='image_id', how='left').fillna(0)
    df_wide['No Finding'] = (df_wide[DISEASES].sum(axis=1) == 0).astype(int)
    for label in FINAL_LABEL_SET:
        if label not in df_wide.columns:
            df_wide[label] = 0
    df_mapped = df_wide[['image_id'] + FINAL_LABEL_SET].copy()
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
    for disease in FINAL_LABEL_SET:
        # Xử lý trường hợp đặc biệt của Pleural Effusion
        if disease == 'Pleural Effusion':
            df_raw[disease] = df_raw['labels'].apply(lambda x: 1 if 'Effusion' in x else 0)
        else:
            df_raw[disease] = df_raw['labels'].apply(lambda x: 1 if disease in x else 0)
            
    # Tạo cột 'No Finding'
    df_raw['No Finding'] = df_raw['labels'].apply(lambda x: 1 if 'No Finding' in x else 0)
    
    # Giữ lại các cột cần thiết
    df_mapped = df_raw[['Image Index'] + FINAL_LABEL_SET].copy()
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
    padchest_common_diseases = [d.lower() for d in FINAL_LABEL_SET]
    
    # Lọc các hàng có nhãn liên quan
    # 'Labels' là cột chứa một list các string
    def filter_labels(label_list):
        if isinstance(label_list, str):
            # Chuyển string dạng list thành list thực sự
            label_list = eval(label_list)
        return any(label in padchest_common_diseases for label in label_list)
    
    df_filtered = df_raw[df_raw['Labels'].apply(filter_labels)].copy()
    
    # Tạo các cột nhị phân cho mỗi bệnh
    for disease_common, disease_padchest in zip(FINAL_LABEL_SET, padchest_common_diseases):
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
                       df_filtered[['ImageID'] + FINAL_LABEL_SET], 
                       on='ImageID', 
                       how='left').fillna(0)
    df_wide = df_wide.drop_duplicates(subset=['ImageID'])
                       
    df_mapped = df_wide[['ImageID'] + FINAL_LABEL_SET].copy()
    df_mapped.rename(columns={'ImageID': 'image_id'}, inplace=True)

    return df_mapped