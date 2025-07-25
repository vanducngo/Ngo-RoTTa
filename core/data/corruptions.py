import torch
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter
from io import BytesIO
from scipy.ndimage import map_coordinates

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

def impulse_noise(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    """Thêm nhiễu Salt & Pepper vào ảnh tensor."""
    c_levels = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
    amount = _linear_interpolate(severity, c_levels)
    if amount == 0: return image_tensor

    # Tạo mặt nạ salt
    salt_mask = torch.rand_like(image_tensor) < (amount / 2.0)
    # Tạo mặt nạ pepper
    pepper_mask = torch.rand_like(image_tensor) < (amount / 2.0)
    
    out = image_tensor.clone()
    out[salt_mask] = 1.0
    out[pepper_mask] = 0.0
    return out

def elastic_transform(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    """Áp dụng biến dạng đàn hồi."""
    c_alpha = [0, 244, 16, 24, 32, 40] 
    c_sigma = [0, 4, 5, 6, 7, 8]
    alpha = _linear_interpolate(severity, c_alpha)
    sigma = _linear_interpolate(severity, c_sigma)
    if alpha == 0: return image_tensor

    # Chuyển sang numpy để xử lý, vì scipy hoạt động trên numpy
    # Giữ nguyên kiểu float, không chuyển sang uint8
    image_np = image_tensor.permute(1, 2, 0).numpy()
    shape = image_np.shape
    
    # Tạo trường dịch chuyển ngẫu nhiên
    dx = gaussian_filter( (np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter( (np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx) # Không dịch chuyển kênh màu

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z+dz, (-1, 1))

    # Áp dụng biến dạng
    distorted_np = map_coordinates(image_np, indices, order=1, mode='reflect').reshape(shape)
    
    return torch.from_numpy(distorted_np).permute(2, 0, 1)

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
    
@_tensor_to_pil_to_tensor
def zoom_blur(image: Image.Image, severity: float = 1) -> Image.Image:
    """Làm mờ do zoom."""
    c_levels = [1.0, 1.10, 1.15, 1.20, 1.25, 1.30]
    zoom_factor = _linear_interpolate(severity, c_levels)
    if zoom_factor == 1.0: return image

    w, h = image.size
    out = image.copy()
    
    for i in range(3): # Zoom 3 lần và lấy trung bình
        zoom_i = 1.0 + (zoom_factor - 1.0) * (i+1)/4.0
        w_i, h_i = int(w/zoom_i), int(h/zoom_i)
        
        img_zoom = image.resize((w_i, h_i), Image.BICUBIC)
        img_crop = img_zoom.crop((
            (w_i-w)//2, (h_i-h)//2,
            (w_i-w)//2 + w, (h_i-h)//2 + h
        ))
        
        if i == 0:
            out = img_crop
        else:
            out = Image.blend(out, img_crop, 1.0/(i+1))
            
    return out

@_tensor_to_pil_to_tensor
def glass_blur(image: Image.Image, severity: float = 1) -> Image.Image:
    """Hiệu ứng nhìn qua kính mờ."""
    c_sigma = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
    c_max_delta = [0, 1, 1, 2, 2, 3]
    c_iterations = [1, 1, 1, 1, 2, 2]
    
    sigma = _linear_interpolate(severity, c_sigma)
    max_delta = int(_linear_interpolate(severity, c_max_delta))
    iterations = int(_linear_interpolate(severity, c_iterations))
    if sigma == 0: return image
        
    image_np = np.array(image, dtype=np.float32) / 255.0
    
    for i in range(iterations):
        # Chọn các vị trí ngẫu nhiên để dịch chuyển pixel
        dx = np.random.randint(-max_delta, max_delta, size=image_np.shape[:2])
        dy = np.random.randint(-max_delta, max_delta, size=image_np.shape[:2])
        x = np.arange(image_np.shape[1])
        y = np.arange(image_np.shape[0])
        xv, yv = np.meshgrid(x, y)
        
        # Áp dụng dịch chuyển
        image_np = image_np[np.clip(yv + dy, 0, image_np.shape[0]-1),
                          np.clip(xv + dx, 0, image_np.shape[1]-1)]
        
        # Làm mờ một chút
        image_np = gaussian_filter(image_np, sigma=sigma)
        
    return Image.fromarray((np.clip(image_np, 0, 1) * 255).astype(np.uint8))

@_tensor_to_pil_to_tensor
def defocus_blur(image: Image.Image, severity: float = 1) -> Image.Image:
    """Làm mờ do mất nét."""
    c_radius = [0, 0.5, 1, 1.5, 2, 2.5]
    radius = _linear_interpolate(severity, c_radius)
    if radius == 0: return image
    return image.filter(ImageFilter.BoxBlur(radius=int(radius)))

# --- Các hàm phức tạp, thường phải dùng PIL để blend ảnh ---
# Các hàm này cần có các ảnh pattern (frost.png, snow.png)
# Tôi sẽ mô phỏng logic mà không cần file ngoài

@_tensor_to_pil_to_tensor
def frost(image: Image.Image, severity: float = 1) -> Image.Image:
    """Hiệu ứng đóng băng."""
    # Logic: Blend ảnh gốc với một ảnh pattern "băng"
    # Ở đây ta tạo pattern băng một cách đơn giản
    w, h = image.size
    frost_pattern = Image.effect_noise((w, h), 128).convert("L").filter(ImageFilter.FIND_EDGES)
    frost_pattern = frost_pattern.convert("RGB")
    
    c_levels = [0, 0.1, 0.15, 0.2, 0.25, 0.3]
    alpha = _linear_interpolate(severity, c_levels)
    if alpha == 0: return image
        
    return Image.blend(image, frost_pattern, alpha=alpha)

@_tensor_to_pil_to_tensor
def snow(image: Image.Image, severity: float = 1) -> Image.Image:
    """Hiệu ứng tuyết rơi."""
    # Logic: Blend ảnh gốc với một ảnh pattern "tuyết"
    w, h = image.size
    snow_pattern = (np.random.rand(h, w) * 255).astype(np.uint8)
    snow_pattern[snow_pattern < 250] = 0 # Tạo các đốm trắng
    snow_pattern = Image.fromarray(snow_pattern).convert("RGB")
    snow_pattern = snow_pattern.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    c_levels = [0, 0.1, 0.15, 0.2, 0.25, 0.3]
    alpha = _linear_interpolate(severity, c_levels)
    if alpha == 0: return image

    return Image.blend(image, snow_pattern, alpha=alpha)

@_tensor_to_pil_to_tensor
def fog(image: Image.Image, severity: float = 1) -> Image.Image:
    """Hiệu ứng sương mù."""
    w, h = image.size
    # Tạo một lớp sương mù bằng nhiễu Perlin/Simplex (hoặc nhiễu Gaussian mờ)
    fog_pattern_np = gaussian_filter(np.random.randn(h, w) * 0.5, sigma=10)
    fog_pattern_np = (fog_pattern_np - np.min(fog_pattern_np)) / (np.max(fog_pattern_np) - np.min(fog_pattern_np))
    fog_pattern = Image.fromarray((fog_pattern_np*255).astype(np.uint8)).convert("RGB")
    
    c_levels = [0, 0.2, 0.3, 0.4, 0.5, 0.6]
    alpha = _linear_interpolate(severity, c_levels)
    if alpha == 0: return image
        
    return Image.blend(image, fog_pattern, alpha=alpha)

# ==============================================================================
# Dictionary và Hàm điều phối chính
# ==============================================================================
CORRUPTION_FUNCS = {
    'gaussian_noise': gaussian_noise,
    'shot_noise': shot_noise,
    'contrast': contrast,
    'brightness': brightness,
    'impulse_noise': impulse_noise,
    'elastic_transform': elastic_transform,
    
    # Các hàm cần PIL
    'motion_blur': motion_blur,
    'zoom_blur': zoom_blur,
    'glass_blur': glass_blur,
    'defocus_blur': defocus_blur,
    'pixelate': pixelate,
    'jpeg_compression': jpeg_compression,
    'frost': frost,
    'snow': snow,
    'fog': fog,
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