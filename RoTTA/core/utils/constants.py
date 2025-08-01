import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CPU_DEVICE = False
# IS_CPU_DEVICE = True


COMMON_DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 
    'Pleural Effusion', 'Pneumothorax'
]
COMMON_FINAL_LABEL_SET = COMMON_DISEASES
TRAINING_LABEL_SET = COMMON_FINAL_LABEL_SET
# TRAINING_LABEL_SET = [
#     'Atelectasis',
#     'Cardiomegaly',
#     'Consolidation',
#     'Pleural Effusion',
#     'Pneumothorax',
#     'Enlarged Cardiomediastinum',
#     'Lung Opacity',
#     'Lung Lesion',
#     'Edema',
#     'Pneumonia',
#     'Pleural Other',
#     'Fracture',
#     'Support Devices',
#     'No Finding',
# ]