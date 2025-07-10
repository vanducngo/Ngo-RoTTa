import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CPU_DEVICE = False