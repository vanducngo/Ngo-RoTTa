COMMON_DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 
    'Pleural Effusion', 'Pneumothorax'
]
COMMON_FINAL_LABEL_SET = ['No Finding'] + COMMON_DISEASES

TRAINING_LABEL_SET = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]

TARGET_TO_FULL_MAP = {
    target_idx: TRAINING_LABEL_SET.index(class_name)
    for target_idx, class_name in enumerate(COMMON_FINAL_LABEL_SET)
}

# Tạo ra một danh sách các chỉ số để dễ dàng slicing tensor
TARGET_INDICES_IN_FULL_LIST = [
    TRAINING_LABEL_SET.index(class_name)
    for class_name in COMMON_FINAL_LABEL_SET
]