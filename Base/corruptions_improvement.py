import torch
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
from scipy.ndimage import map_coordinates

def _linear_interpolate(value: float, points: list):
    lower_pt = int(np.floor(value))
    upper_pt = int(np.ceil(value))
    
    if lower_pt == upper_pt:
        return points[lower_pt]

    weight = value - lower_pt
    return (1 - weight) * points[lower_pt] + weight * points[upper_pt]

def _apply_gaussian_blur(image_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma < 0.1: 
        return image_tensor
        
    channels = image_tensor.shape[0]
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0: kernel_size += 1
    
    x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=image_tensor.device)
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    kernel_2d = kernel_1d.view(1, 1, 1, kernel_size) * kernel_1d.view(1, 1, kernel_size, 1)
    kernel = kernel_2d.repeat(channels, 1, 1, 1)
    
    return F.conv2d(image_tensor.unsqueeze(0), kernel, padding=kernel_size // 2, groups=channels).squeeze(0)

# ==============================================================================
#  Add noise with Tensor-native
# ==============================================================================

def gaussian_noise(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    c_levels = [0, 0.04, 0.06, 0.08, 0.09, 0.10]
    scale = _linear_interpolate(severity, c_levels)
    if scale == 0: return image_tensor
    noise = torch.randn_like(image_tensor) * scale
    return torch.clamp(image_tensor + noise, 0, 1)

def shot_noise(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    c_levels = [float('inf'), 500, 250, 100, 75, 50]
    scale = _linear_interpolate(severity, c_levels)
    if scale == float('inf'): 
        return image_tensor

    device = image_tensor.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    # --- Step 1: UN-NORMALIZE ---
    image_unnormalized = image_tensor * std + mean
    image_unnormalized = torch.clamp(image_unnormalized, 0, 1)

    # --- Step 2: Apply noise in the data [0, 1] ---
    corrupted_unnormalized = torch.clamp(torch.poisson(image_unnormalized * scale) / scale, 0, 1)

    # --- BƯỚC 3: RE-NORMALIZE ---
    corrupted_normalized = (corrupted_unnormalized - mean) / std

    return corrupted_normalized

def contrast(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    c_levels = [1.0, 0.75, 0.5, 0.4, 0.3, 0.2]
    scale = _linear_interpolate(severity, c_levels)
    if scale == 1.0: return image_tensor
    mean = torch.mean(image_tensor, dim=[-2, -1], keepdim=True)
    return torch.clamp((image_tensor - mean) * scale + mean, 0, 1)

def brightness(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    c_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    scale = _linear_interpolate(severity, c_levels)
    if scale == 0: return image_tensor
    return torch.clamp(image_tensor + scale, 0, 1)

def impulse_noise(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    c_levels = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
    amount = _linear_interpolate(severity, c_levels)
    if amount == 0: return image_tensor
    salt_mask = torch.rand_like(image_tensor) < (amount / 2.0)
    pepper_mask = torch.rand_like(image_tensor) < (amount / 2.0)
    out = image_tensor.clone()
    out[salt_mask] = 1.0
    out[pepper_mask] = 0.0
    return out

def elastic_transform(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    c_alpha = [0, 244, 16, 24, 32, 40] 
    c_sigma = [0, 4, 5, 6, 7, 8]
    alpha = _linear_interpolate(severity, c_alpha)
    sigma = _linear_interpolate(severity, c_sigma)
    if alpha == 0: return image_tensor
    
    image_np = image_tensor.permute(1, 2, 0).numpy()
    shape = image_np.shape
    
    dx = scipy_gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = scipy_gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z+dz, (-1, 1))
    
    distorted_np = map_coordinates(image_np, indices, order=1, mode='reflect').reshape(shape)
    return torch.from_numpy(distorted_np).permute(2, 0, 1)

def motion_blur(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    c_levels = [1, 7, 9, 13, 15, 21]
    kernel_size = int(round(_linear_interpolate(severity, c_levels)))
    if kernel_size % 2 == 0: kernel_size += 1
    if kernel_size <= 1: return image_tensor

    channels = image_tensor.shape[0]
    kernel = torch.zeros(channels, 1, kernel_size, kernel_size, device=image_tensor.device)
    kernel[:, 0, kernel_size // 2, :] = 1.0 / kernel_size
    return F.conv2d(image_tensor.unsqueeze(0), kernel, padding=(kernel_size // 2, kernel_size // 2), groups=channels).squeeze(0)

def pixelate(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    c_levels = [1.0, 0.88, 0.75, 0.6, 0.5, 0.4]
    scale = _linear_interpolate(severity, c_levels)
    if scale == 1.0: return image_tensor

    _, h, w = image_tensor.shape
    small_size = (int(h * scale), int(w * scale))
    small = F.interpolate(image_tensor.unsqueeze(0), size=small_size, mode='bilinear', align_corners=False).squeeze(0)
    return F.interpolate(small.unsqueeze(0), size=(h, w), mode='nearest').squeeze(0)

def jpeg_compression(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    c_levels = [256, 64, 32, 16, 8, 4]
    levels = int(_linear_interpolate(severity, c_levels))
    if levels >= 256: return image_tensor
    return torch.round(image_tensor * (levels - 1)) / (levels - 1)

def gaussian_blur(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    c_levels = [0, 0.5, 1, 1.5, 2, 2.5]
    sigma = _linear_interpolate(severity, c_levels)
    return _apply_gaussian_blur(image_tensor, sigma)

def zoom_blur(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    c_levels = [1.0, 1.10, 1.15, 1.20, 1.25, 1.30]
    zoom_factor = _linear_interpolate(severity, c_levels)
    if zoom_factor == 1.0: return image_tensor

    _, h, w = image_tensor.shape
    out = torch.zeros_like(image_tensor)
    for i in range(4):
        zoom_i = 1.0 + (zoom_factor - 1.0) * (i + 1) / 4.0
        new_h, new_w = int(h / zoom_i), int(w / zoom_i)
        
        zoomed = F.interpolate(image_tensor.unsqueeze(0), size=(new_h, new_w), mode='bicubic', align_corners=False).squeeze(0)
        
        # Pad to original size before adding
        pad_h, pad_w = h - new_h, w - new_w
        top, left = pad_h // 2, pad_w // 2
        padded = F.pad(zoomed, (left, pad_w - left, top, pad_h - top))
        out += padded
        
    return torch.clamp(out / 4, 0, 1)

def glass_blur(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    c_sigma = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
    c_max_delta = [0, 1, 1, 2, 2, 3]
    c_iterations = [1, 1, 1, 1, 2, 2]
    sigma = _linear_interpolate(severity, c_sigma)
    max_delta = int(round(_linear_interpolate(severity, c_max_delta)))
    iterations = int(round(_linear_interpolate(severity, c_iterations)))
    if max_delta == 0: return image_tensor

    _, h, w = image_tensor.shape
    
    # Grid sample requires 4D input
    image_batch = image_tensor.unsqueeze(0)
    
    for _ in range(iterations):
        dx = torch.randint(-max_delta, max_delta + 1, (1, h, w, 1), device=image_tensor.device).float()
        dy = torch.randint(-max_delta, max_delta + 1, (1, h, w, 1), device=image_tensor.device).float()
        
        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=image_tensor.device), torch.arange(w, device=image_tensor.device), indexing='ij')
        
        grid = torch.stack([grid_x, grid_y], dim=-1).float().unsqueeze(0)
        grid = grid + torch.cat([dx, dy], dim=-1)
        
        # Normalize grid to [-1, 1]
        grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / (w - 1) - 1.0
        grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / (h - 1) - 1.0
        
        image_batch = F.grid_sample(image_batch, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
        image_batch = _apply_gaussian_blur(image_batch.squeeze(0), sigma).unsqueeze(0)
        
    return torch.clamp(image_batch.squeeze(0), 0, 1)

def defocus_blur(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    c_radius = [0, 0.5, 1, 1.5, 2, 2.5]
    radius = _linear_interpolate(severity, c_radius)
    if radius < 0.1: return image_tensor

    kernel_size = int(2 * radius + 1)
    if kernel_size % 2 == 0: kernel_size += 1

    channels = image_tensor.shape[0]
    kernel = torch.ones(channels, 1, kernel_size, kernel_size, device=image_tensor.device) / (kernel_size ** 2)
    return F.conv2d(image_tensor.unsqueeze(0), kernel, padding=kernel_size // 2, groups=channels).squeeze(0)

def frost(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    c_levels = [0, 0.1, 0.15, 0.2, 0.25, 0.3]
    alpha = _linear_interpolate(severity, c_levels)
    if alpha == 0: return image_tensor

    _, h, w = image_tensor.shape
    noise = torch.randn(h, w, device=image_tensor.device)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image_tensor.device, dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    edge_x = F.conv2d(noise.unsqueeze(0).unsqueeze(0), sobel_x, padding=1).squeeze(0).squeeze(0)
    edge_y = F.conv2d(noise.unsqueeze(0).unsqueeze(0), sobel_y, padding=1).squeeze(0).squeeze(0)
    frost_pattern = torch.sqrt(edge_x**2 + edge_y**2).unsqueeze(0).repeat(image_tensor.shape[0], 1, 1)
    frost_pattern = (frost_pattern - frost_pattern.min()) / (frost_pattern.max() - frost_pattern.min())
    return torch.clamp((1 - alpha) * image_tensor + alpha * frost_pattern, 0, 1)

# def snow(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
#     c_levels = [0, 0.1, 0.15, 0.2, 0.25, 0.3]
#     alpha = _linear_interpolate(severity, c_levels)
#     if alpha == 0: return image_tensor

#     snow_pattern = torch.rand_like(image_tensor) * 0.7 # Less intense
#     snow_pattern = (snow_pattern > 0.995).float() # Sparse flakes
#     snow_pattern = _apply_gaussian_blur(snow_pattern, sigma=1.5)
    
#     # Whiten and brighten flakes
#     snow_pattern = (snow_pattern - snow_pattern.min()) / (snow_pattern.max() - snow_pattern.min() + 1e-6)
    
#     return torch.clamp(image_tensor + snow_pattern * alpha, 0, 1)

def snow(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    """Thêm hiệu ứng tuyết rơi lấm tấm, mô phỏng các bông tuyết."""
    alpha_levels = [0, 0.2, 0.3, 0.4, 0.5, 0.6] 
    flake_size_levels = [1.0, 0.998, 0.996, 0.994, 0.992, 0.99]
    blur_sigma_levels = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
    
    alpha = _linear_interpolate(severity, alpha_levels)
    if alpha == 0: return image_tensor
        
    flake_threshold = _linear_interpolate(severity, flake_size_levels)
    blur_sigma = _linear_interpolate(severity, blur_sigma_levels)

    c, h, w = image_tensor.shape
    
    # --- SỬA LỖI: TẠO NHIỄU TRẮNG ---
    # 1. Tạo nhiễu cho 1 kênh duy nhất
    snow_pattern_1ch = (torch.rand(1, h, w, device=image_tensor.device) > flake_threshold).float()
    
    # 2. Làm mờ kênh duy nhất đó
    if blur_sigma > 0:
        snow_pattern_1ch = _apply_gaussian_blur(snow_pattern_1ch, sigma=blur_sigma)

    # 3. Lặp lại (repeat) kênh đã mờ cho cả 3 kênh màu
    snow_mask = snow_pattern_1ch.repeat(c, 1, 1)
    # --- KẾT THÚC SỬA LỖI ---
    
    snow_mask = (snow_mask - snow_mask.min()) / (snow_mask.max() - snow_mask.min() + 1e-6)
    snow_layer = torch.ones_like(image_tensor)
    effective_mask = snow_mask * alpha
    
    corrupted_image = image_tensor * (1 - effective_mask) + snow_layer * effective_mask
    
    return torch.clamp(corrupted_image, 0, 1)

def fog(image_tensor: torch.Tensor, severity: float = 1) -> torch.Tensor:
    """
    Thêm hiệu ứng sương mù vào một batch ảnh.
    Phiên bản này được tối ưu hóa bằng cách tạo nhiễu tần số thấp
    thông qua việc phóng to một pattern nhiễu nhỏ.

    Args:
        image_tensor (torch.Tensor): Batch ảnh đầu vào, shape [B, C, H, W].
        severity (float): Mức độ hiệu ứng, từ 0 đến 5.

    Returns:
        torch.Tensor: Batch ảnh đã được thêm hiệu ứng sương mù.
    """
    # 1. Tính toán các tham số dựa trên severity
    # alpha: độ mờ đục của lớp sương mù
    # scale_factor: kích thước của pattern nhiễu ban đầu (càng nhỏ, sương mù càng "dày")
    alpha_levels = [0, 0.15, 0.25, 0.35, 0.45, 0.55] # Điều chỉnh alpha để hiệu ứng rõ hơn
    scale_levels = [1, 16, 14, 12, 10, 8] # Mẫu số để chia (ví dụ: h/16)
    
    alpha = _linear_interpolate(severity, alpha_levels)
    if alpha == 0: 
        return image_tensor
        
    scale_divisor = _linear_interpolate(severity, scale_levels)
    
    # Lấy kích thước của batch
    b, c, h, w = image_tensor.shape
    device = image_tensor.device
    
    # 2. Tạo một tensor nhiễu rất nhỏ
    # Kích thước nhỏ hơn sẽ tạo ra các đám sương mù "lớn" hơn, "dày" hơn
    small_h, small_w = max(1, int(h / scale_divisor)), max(1, int(w / scale_divisor))
    
    # Tạo nhiễu cho cả batch, nhưng chỉ 1 kênh
    fog_pattern_small = torch.randn(b, 1, small_h, small_w, device=device)

    # 3. Phóng to (interpolate) pattern nhiễu lên kích thước đầy đủ
    # Quá trình này sẽ tự động làm "mịn" nhiễu, tạo ra hiệu ứng mượt mà
    # 'bicubic' cho kết quả mịn hơn 'bilinear'
    fog_pattern_large = F.interpolate(fog_pattern_small, size=(h, w), mode='bicubic', align_corners=False)
    
    # 4. Chuẩn hóa pattern về khoảng [0, 1]
    # Thực hiện trên từng ảnh trong batch một cách độc lập
    # reshape(-1) -> tính min/max trên toàn bộ ảnh -> reshape lại
    fog_pattern_flat = fog_pattern_large.view(b, -1)
    min_vals = fog_pattern_flat.min(dim=1, keepdim=True)[0]
    max_vals = fog_pattern_flat.max(dim=1, keepdim=True)[0]
    fog_pattern_normalized = (fog_pattern_large.view(b, -1) - min_vals) / (max_vals - min_vals + 1e-6)
    fog_pattern_normalized = fog_pattern_normalized.view(b, 1, h, w)

    # 5. Lặp lại pattern cho tất cả các kênh màu (ví dụ 3 kênh R,G,B)
    fog_pattern_batch = fog_pattern_normalized.repeat(1, c, 1, 1)
    
    # 6. Trộn ảnh gốc với lớp sương mù
    # Công thức: new_image = (1 - alpha) * original_image + alpha * fog_pattern
    corrupted_image = (1 - alpha) * image_tensor + alpha * fog_pattern_batch
    
    return torch.clamp(corrupted_image, 0, 1)


# ==============================================================================
# Dictionary và fuctions
# ==============================================================================
CORRUPTION_FUNCS = {
    'gaussian_noise': gaussian_noise, 
    'shot_noise': shot_noise, 
    'impulse_noise': impulse_noise,
    'defocus_blur': defocus_blur, 
    'glass_blur': glass_blur, 
    'motion_blur': motion_blur, 
    'zoom_blur': zoom_blur,
    'snow': snow, 
    'frost': frost, 
    'fog': fog, 
    'brightness': brightness,
    'contrast': contrast, 
    'elastic_transform': elastic_transform, 
    'pixelate': pixelate, 
    'jpeg_compression': jpeg_compression,
    'gaussian_blur': gaussian_blur
}

BATCH_CORRUPTION_FUNCS = ['fog'] 
def apply_corruption(image_tensor: torch.Tensor, corruption_name: str, severity: float = 1) -> torch.Tensor:
    if severity == 0 or corruption_name.lower() == 'none':
        return image_tensor
    if not (0 < severity <= 5):
        raise ValueError(f"Severity must be between 0 (exclusive) and 5 (inclusive), but got {severity}")
    if corruption_name not in CORRUPTION_FUNCS:
        raise ValueError(f"Unknown corruption type: {corruption_name}")

    original_device = image_tensor.device
    corruption_func = CORRUPTION_FUNCS[corruption_name]

    # --- LOGIC MỚI ĐỂ XỬ LÝ BATCH HOẶC SINGLE IMAGE ---
    
    # 1. Nếu hàm nhiễu được thiết kế để xử lý cả batch
    if corruption_name in BATCH_CORRUPTION_FUNCS:
        # Và input là một batch
        if image_tensor.dim() == 4:
            # Truyền thẳng cả batch vào
            return corruption_func(image_tensor, severity)
        # Nếu input là ảnh đơn, vẫn phải unsqueeze để nó thành batch 1 ảnh
        elif image_tensor.dim() == 3:
            return corruption_func(image_tensor.unsqueeze(0), severity).squeeze(0)

    # 2. Nếu là hàm nhiễu thông thường (chỉ xử lý ảnh đơn)
    else:
        image_tensor_cpu = image_tensor.cpu()
        if image_tensor_cpu.dim() == 4: # Batch
            corrupted_images = [corruption_func(img, severity) for img in image_tensor_cpu]
            return torch.stack(corrupted_images).to(original_device)
        elif image_tensor_cpu.dim() == 3: # Single image
            return corruption_func(image_tensor_cpu, severity).to(original_device)
        else:
            raise ValueError("Input tensor must have 3 or 4 dimensions")
