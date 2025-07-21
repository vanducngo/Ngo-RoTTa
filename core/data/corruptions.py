import torch
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
from io import BytesIO

# ==============================================================================
# Hàm điều phối chính (Dispatcher Function)
# ==============================================================================

def apply_corruption(image_tensor: torch.Tensor, corruption_name: str, severity: int = 1) -> torch.Tensor:
    """
    Áp dụng một loại nhiễu cụ thể lên một batch hoặc một ảnh tensor.
    
    Args:
        image_tensor (torch.Tensor): Tensor ảnh có shape (C, H, W) hoặc (B, C, H, W).
                                     Giá trị pixel được giả định trong khoảng [0, 1].
        corruption_name (str): Tên của loại nhiễu cần áp dụng.
        severity (int): Mức độ nhiễu, từ 1 đến 5.

    Returns:
        torch.Tensor: Tensor ảnh đã bị làm nhiễu.
    """

    if severity == 0 or corruption_name.lower() == 'none':
        return image_tensor

    if corruption_name not in CORRUPTION_FUNCS:
        raise ValueError(f"Unknown corruption type: {corruption_name}")
    
    if not (1 <= severity <= 5):
        raise ValueError(f"Severity must be between 1 and 5, but got {severity}")

    # Hàm tạo nhiễu sẽ được gọi
    corruption_func = CORRUPTION_FUNCS[corruption_name]
    
    # Chuyển tensor về ảnh PIL để xử lý
    # Xử lý cả batch hoặc ảnh đơn
    if image_tensor.dim() == 4: # Batch of images (B, C, H, W)
        corrupted_images = []
        for img in image_tensor:
            pil_img = TF.to_pil_image(img)
            corrupted_pil = corruption_func(pil_img, severity)
            corrupted_images.append(TF.to_tensor(corrupted_pil))
        return torch.stack(corrupted_images)
    elif image_tensor.dim() == 3: # Single image (C, H, W)
        pil_img = TF.to_pil_image(image_tensor)
        corrupted_pil = corruption_func(pil_img, severity)
        return TF.to_tensor(corrupted_pil)
    else:
        raise ValueError("Input tensor must have 3 or 4 dimensions")


# ==============================================================================
# Các hàm tạo nhiễu cụ thể
# Mỗi hàm nhận một ảnh PIL và severity, trả về một ảnh PIL.
# ==============================================================================

def gaussian_noise(image: Image.Image, severity: int = 1) -> Image.Image:
    """Thêm nhiễu Gaussian vào ảnh."""
    c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]
    image_np = np.array(image) / 255.
    noise = np.random.normal(size=image_np.shape, scale=c)
    image_noisy_np = np.clip(image_np + noise, 0, 1)
    return Image.fromarray((image_noisy_np * 255).astype(np.uint8))

def shot_noise(image: Image.Image, severity: int = 1) -> Image.Image:
    """Thêm nhiễu Shot (Poisson) vào ảnh."""
    c = [500, 250, 100, 75, 50][severity - 1]
    image_np = np.array(image) / 255.
    image_noisy_np = np.clip(np.random.poisson(image_np * c) / c, 0, 1)
    return Image.fromarray((image_noisy_np * 255).astype(np.uint8))

def contrast(image: Image.Image, severity: int = 1) -> Image.Image:
    """Thay đổi độ tương phản của ảnh."""
    c = [0.75, 0.5, 0.4, 0.3, 0.2][severity - 1]
    image_np = np.array(image) / 255.
    mean = np.mean(image_np, axis=(0, 1), keepdims=True)
    image_contrast_np = np.clip((image_np - mean) * c + mean, 0, 1)
    return Image.fromarray((image_contrast_np * 255).astype(np.uint8))

def motion_blur(image: Image.Image, severity: int = 1) -> Image.Image:
    """Làm mờ do chuyển động."""
    # Thay đổi các kernel_size chẵn thành số lẻ gần nhất
    c = [(7, 1), (9, 1.5), (13, 2), (15, 2.5), (21, 3)][severity - 1]    
    kernel_size, sigma = c

    # Logic tạo kernel bây giờ sẽ hoạt động đúng
    kernel = np.zeros((kernel_size, kernel_size))
    # Ví dụ với kernel_size=7, tâm là int((7-1)/2) = 3 (đúng)
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size

     # --- THÊM PRINT DEBUG ---
    # In ra giá trị kernel_size ngay trước khi sử dụng
    print(f"[DEBUG] Using kernel_size: {kernel_size}")
    # --- KẾT THÚC THÊM PRINT DEBUG ---
    
    # Kiểm tra rõ ràng một lần nữa
    if kernel_size % 2 == 0:
        print("[DEBUG] FATAL: kernel_size is EVEN! Raising error manually.")
        raise ValueError(f"Kernel size must be odd, but got {kernel_size}")
    
    return image.filter(ImageFilter.Kernel((kernel_size, kernel_size), kernel.flatten()))

def pixelate(image: Image.Image, severity: int = 1) -> Image.Image:
    """Làm vỡ ảnh (pixelate)."""
    c = [0.88, 0.75, 0.6, 0.5, 0.4][severity - 1]
    w, h = image.size
    # Giảm kích thước ảnh rồi phóng to lại
    image_small = image.resize((int(w * c), int(h * c)), Image.BOX)
    return image_small.resize(image.size, Image.BOX)

def jpeg_compression(image: Image.Image, severity: int = 1) -> Image.Image:
    """Nén ảnh theo chuẩn JPEG."""
    c = [40, 30, 20, 15, 10][severity - 1]
    output = BytesIO()
    image.save(output, 'JPEG', quality=c)
    return Image.open(output)
    
def brightness(image: Image.Image, severity: int = 1) -> Image.Image:
    """Thay đổi độ sáng."""
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    image_np = np.array(image) / 255.
    image_bright_np = np.clip(image_np + c, 0, 1)
    return Image.fromarray((image_bright_np * 255).astype(np.uint8))

def gaussian_blur(image: Image.Image, severity: int = 1) -> Image.Image:
    """Làm mờ Gaussian."""
    c = [0.5, 1, 1.5, 2, 2.5][severity - 1]
    return image.filter(ImageFilter.GaussianBlur(radius=c))
    
# ==============================================================================
# Dictionary để map tên nhiễu với hàm tương ứng
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
    # Thêm các loại nhiễu khác vào đây nếu cần
}