COMMON_DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 
    'Pleural Effusion', 'Pneumothorax'
]

COMMON_FINAL_LABEL_SET = COMMON_DISEASES
TRAINING_LABEL_SET = COMMON_FINAL_LABEL_SET

TARGET_INDICES_IN_FULL_LIST = [
    TRAINING_LABEL_SET.index(class_name)
    for class_name in COMMON_FINAL_LABEL_SET
]