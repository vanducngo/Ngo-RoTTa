import torch
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
from io import BytesIO

# ==============================================================================
# Hàm Helper cho Nội suy tuyến tính
# ==============================================================================

def _linear_interpolate(value: float, points: list):
    """
    Nội suy tuyến tính.
    `points` là một danh sách các giá trị tại các điểm nguyên 0, 1, 2, ...
    """
    lower_pt = int(np.floor(value))
    upper_pt = int(np.ceil(value))
    
    if lower_pt == upper_pt:
        return points[lower_pt]

    weight = value - lower_pt
    return (1 - weight) * points[lower_pt] + weight * points[upper_pt]

# ==============================================================================
# Các hàm tạo nhiễu - Đa số hoạt động trên TENSOR
# ==============================================================================

# --- Các hàm hoạt động trực tiếp trên Tensor ---

def gaussian_noise(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    """Thêm nhiễu Gaussian vào ảnh tensor [0, 1]."""
    c_levels = [0, 0.04, 0.06, 0.08, 0.09, 0.10] # Thêm 0 cho severity=0
    scale = _linear_interpolate(severity, c_levels)
    if scale == 0: return image_tensor
    
    noise = torch.randn_like(image_tensor) * scale
    return torch.clamp(image_tensor + noise, 0, 1)

def shot_noise(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    """Thêm nhiễu Shot (Poisson) vào ảnh tensor [0, 1]."""
    c_levels = [float('inf'), 500, 250, 100, 75, 50] # inf cho severity=0 (ko nhiễu)
    scale = _linear_interpolate(severity, c_levels)
    if scale == float('inf'): return image_tensor

    # Poisson noise phụ thuộc vào giá trị pixel, cần lặp
    return torch.clamp(torch.poisson(image_tensor * scale) / scale, 0, 1)

def contrast(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    """Thay đổi độ tương phản của ảnh tensor [0, 1]."""
    c_levels = [1.0, 0.75, 0.5, 0.4, 0.3, 0.2] # 1.0 cho severity=0
    scale = _linear_interpolate(severity, c_levels)
    if scale == 1.0: return image_tensor

    mean = torch.mean(image_tensor, dim=[-2, -1], keepdim=True)
    return torch.clamp((image_tensor - mean) * scale + mean, 0, 1)

def brightness(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    """Thay đổi độ sáng của ảnh tensor [0, 1]."""
    c_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5] # 0 cho severity=0
    scale = _linear_interpolate(severity, c_levels)
    if scale == 0: return image_tensor

    return torch.clamp(image_tensor + scale, 0, 1)

# --- Các hàm yêu cầu chuyển đổi sang PIL ---

def _tensor_to_pil_to_tensor(corruption_func):
    """Decorator để xử lý chuyển đổi qua lại cho các hàm cần PIL.Image."""
    def wrapper(image_tensor: torch.Tensor, severity: float = 1):
        if image_tensor.dim() != 3:
            raise TypeError("This function only accepts single image tensors (C, H, W)")
        
        pil_img = TF.to_pil_image(image_tensor)
        corrupted_pil = corruption_func(pil_img, severity)
        return TF.to_tensor(corrupted_pil)
    return wrapper

@_tensor_to_pil_to_tensor
def motion_blur(image: Image.Image, severity: float = 1) -> Image.Image:
    """Làm mờ do chuyển động."""
    k_levels = [1, 7, 9, 13, 15, 21]
    interpolated_k = _linear_interpolate(severity, k_levels)
    kernel_size = int(round(interpolated_k))
    if kernel_size % 2 == 0: kernel_size += 1
    if kernel_size <= 1: return image
        
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[int((kernel_size - 1) / 2), :] = 1.0
    kernel = kernel / np.sum(kernel)
    return image.filter(ImageFilter.Kernel((kernel_size, kernel_size), kernel.flatten().tolist()))

@_tensor_to_pil_to_tensor
def pixelate(image: Image.Image, severity: float = 1) -> Image.Image:
    """Làm vỡ ảnh (pixelate)."""
    c_levels = [1.0, 0.88, 0.75, 0.6, 0.5, 0.4] # 1.0 cho severity=0
    scale = _linear_interpolate(severity, c_levels)
    if scale == 1.0: return image

    w, h = image.size
    image_small = image.resize((int(w * scale), int(h * scale)), Image.BOX)
    return image_small.resize(image.size, Image.BOX)

@_tensor_to_pil_to_tensor
def jpeg_compression(image: Image.Image, severity: float = 1) -> Image.Image:
    """Nén ảnh theo chuẩn JPEG."""
    c_levels = [100, 40, 30, 20, 15, 10] # 100 cho severity=0
    quality = int(_linear_interpolate(severity, c_levels))
    if quality == 100: return image

    output = BytesIO()
    image.save(output, 'JPEG', quality=quality)
    return Image.open(output)

@_tensor_to_pil_to_tensor
def gaussian_blur(image: Image.Image, severity: float = 1) -> Image.Image:
    """Làm mờ Gaussian."""
    c_levels = [0, 0.5, 1, 1.5, 2, 2.5] # 0 cho severity=0
    radius = _linear_interpolate(severity, c_levels)
    if radius == 0: return image
        
    return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
# ==============================================================================
# Dictionary và Hàm điều phối chính
# ==============================================================================

CORRUPTION_FUNCS = {
    'gaussian_noise': gaussian_noise,
    'shot_noise': shot_noise,
    'contrast': contrast,
    'motion_blur': motion_blur,
    'pixelate': pixelate,
    'jpeg_compression': jpeg_compression,
    'brightness': brightness,
    'gaussian_blur': gaussian_blur
}

def apply_corruption(image_tensor: torch.Tensor, corruption_name: str, severity: float = 1) -> torch.Tensor:
    if severity == 0 or corruption_name.lower() == 'none':
        return image_tensor
    if not (0 < severity <= 5):
        raise ValueError(f"Severity must be between 0 (exclusive) and 5 (inclusive), but got {severity}")
    if corruption_name not in CORRUPTION_FUNCS:
        raise ValueError(f"Unknown corruption type: {corruption_name}")

    original_device = image_tensor.device
    image_tensor_cpu = image_tensor.cpu()
    corruption_func = CORRUPTION_FUNCS[corruption_name]

    if image_tensor_cpu.dim() == 4: # Batch
        corrupted_images = [corruption_func(img, severity) for img in image_tensor_cpu]
        return torch.stack(corrupted_images).to(original_device)
    elif image_tensor_cpu.dim() == 3: # Single image
        return corruption_func(image_tensor_cpu, severity).to(original_device)
    else:
        raise ValueError("Input tensor must have 3 or 4 dimensions")