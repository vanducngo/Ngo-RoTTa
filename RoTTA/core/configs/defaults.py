from yacs.config import CfgNode as CN


_C = CN()
cfg = _C


# ----------------------------- Model options ------------------------------- #
_C.MODEL = CN()

_C.MODEL.ARCH = 'Standard'

_C.MODEL.EPISODIC = False

_C.MODEL.PROJECTION = CN()

_C.MODEL.PROJECTION.HEAD = "linear"
_C.MODEL.PROJECTION.EMB_DIM = 2048
_C.MODEL.PROJECTION.FEA_DIM = 128

# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CN()

_C.CORRUPTION.DATASET = 'cifar10'
_C.CORRUPTION.SOURCE = ''
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']
_C.CORRUPTION.SEVERITY = [5, 4, 3, 2, 1]
_C.CORRUPTION.NUM_EX = 10000
_C.CORRUPTION.NUM_CLASS = -1

# ----------------------------- Input options -------------------------- #
_C.INPUT = CN()

_C.INPUT.SIZE = (32, 32)
_C.INPUT.INTERPOLATION = "bilinear"
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
_C.INPUT.TRANSFORMS = ("normalize", )

# ----------------------------- loader options -------------------------- #
_C.LOADER = CN()

_C.LOADER.SAMPLER = CN()
_C.LOADER.SAMPLER.TYPE = "sequence"
_C.LOADER.SAMPLER.GAMMA = 0.1

_C.LOADER.NUM_WORKS = 2


# ------------------------------- Batch norm options ------------------------ #
_C.BN = CN()
_C.BN.EPS = 1e-5
_C.BN.MOM = 0.1

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CN()
_C.OPTIM.STEPS = 1
_C.OPTIM.LR = 1e-3

_C.OPTIM.METHOD = 'Adam'
_C.OPTIM.BETA = 0.9
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.DAMPENING = 0.0
_C.OPTIM.NESTEROV = True
_C.OPTIM.WD = 0.0
  
# --- Các tham số đặc trưng của CoTTA ---
# Mean Teacher update rate
_C.OPTIM.MT = 0.999

# Stochastic Restore probability
_C.OPTIM.RST = 0.01

# Augmentation-averaging threshold
_C.OPTIM.AP = 0.72

# ------------------------------- Testing options --------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64

# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True

# ---------------------------------- Misc options --------------------------- #

_C.DESC = ""
_C.SEED = 42
_C.OUTPUT_DIR = "./output"
_C.DATA_DIR = "./datasets"
_C.CKPT_DIR = "./ckpt"
_C.LOG_DEST = "log.txt"

_C.LOG_TIME = ''
_C.DEBUG = 0

# tta method specific
_C.ADAPTER = CN()

_C.ADAPTER.NAME = "rotta"

_C.ADAPTER.RoTTA = CN()
_C.ADAPTER.RoTTA.MEMORY_SIZE = 64
_C.ADAPTER.RoTTA.UPDATE_FREQUENCY = 64
_C.ADAPTER.RoTTA.NU = 0.001
_C.ADAPTER.RoTTA.ALPHA = 0.05
_C.ADAPTER.RoTTA.LAMBDA_T = 1.0
_C.ADAPTER.RoTTA.LAMBDA_U = 1.0

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


# Thêm mục DATASET vào đây để yacs nhận diện được nó
_C.DATASET = CN()
_C.DATASET.NAME = 'DefaultDataset'
# Định nghĩa các khóa bạn sẽ dùng, có thể để giá trị mặc định là chuỗi rỗng hoặc list rỗng
_C.DATASET.TEST_DOMAINS = []
_C.DATASET.LABELS_LIST = []

# Định nghĩa các khóa cho đường dẫn, để giá trị mặc định là chuỗi rỗng
# Cần định nghĩa cho TẤT CẢ các domain bạn có thể dùng
_C.DATASET.VINDR_PATH = ""
_C.DATASET.VINDR_CSV = ""
_C.DATASET.VINDR_IMAGE_DIR = ""

_C.DATASET.NIH14_PATH = ""
_C.DATASET.NIH14_CSV = ""
_C.DATASET.NIH14_IMAGE_DIR = ""

_C.DATASET.PADCHEST_PATH = ""
_C.DATASET.PADCHEST_CSV = ""
_C.DATASET.PADCHEST_IMAGE_DIR = ""

_C.DATASET.CHEXPERT_PATH = ""
_C.DATASET.CHEXPERT_CSV = ""
_C.DATASET.CHEXPERT_IMAGE_DIR = ""

_C.DATASET.ADAPTATION_MODE = ""
_C.DATASET.BASE_DOMAIN = CN()
_C.DATASET.BASE_DOMAIN.PATH = ""
_C.DATASET.BASE_DOMAIN.CSV = ""
_C.DATASET.BASE_DOMAIN.IMAGE_DIR = ""

# Danh sách các loại nhiễu sẽ được áp dụng theo thứ tự
_C.DATASET.TEST_CORRUPTIONS = []
# Mức độ nhiễu (1-5)
_C.DATASET.SEVERITY = 0.2

if 'MODEL' not in _C:
    _C.MODEL = CN()
_C.MODEL.NUM_CLASSES = -1 # Giá trị mặc định

# Thêm INPUT nếu nó chưa tồn tại
if 'INPUT' not in _C:
    _C.INPUT = CN()
_C.INPUT.SIZE = (224, 224)
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

_C.OUTPUT_DIR = "Log/unknow"