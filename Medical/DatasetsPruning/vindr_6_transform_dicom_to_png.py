import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import cv2 # Sử dụng OpenCV để xử lý ảnh
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image

VINDR_ROOT_PATH = "/Users/admin/Working/Data/MixData/vinbigdata-chest-xray-30-percent"

OUTPUT_ROOT_PATH = "/Users/admin/Working/Data/vinbigdata_structured"

SETS_TO_CONVERT = {
    'images': 'validate.csv',
    # 'test': 'test.csv' 
}

TARGET_SIZE = 224

def process_dicom_to_png(dicom_path, output_path, target_size=224):
    """
    Đọc một file DICOM, xử lý, thay đổi kích thước và lưu thành file PNG.
    
    Args:
        dicom_path (str): Đường dẫn đến file DICOM gốc.
        output_path (str): Đường dẫn để lưu file PNG.
        target_size (int): Kích thước cạnh của ảnh vuông đầu ra.
    
    Returns:
        bool: True nếu thành công, False nếu thất bại.
    """
    try:
        # 1. Đọc file DICOM
        dicom = pydicom.dcmread(dicom_path)        
        
        # 2. Áp dụng VOI LUT để cải thiện độ tương phản
        pixel_array = apply_voi_lut(dicom.pixel_array, dicom)
        
        # 3. Xử lý ảnh đơn sắc (monochrome) nếu cần
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = np.amax(pixel_array) - pixel_array
            
        # 4. Chuẩn hóa pixel về dải giá trị 8-bit [0, 255]
        pixel_array = pixel_array - np.min(pixel_array)
        pixel_array = pixel_array / (np.max(pixel_array) + 1e-6) # Tránh chia cho 0
        image = (pixel_array * 255).astype(np.uint8)
        
        # 5. Thay đổi kích thước ảnh bằng OpenCV
        # cv2.INTER_AREA thường tốt cho việc thu nhỏ ảnh
        resized_image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        # 6. Lưu ảnh dưới dạng PNG
        cv2.imwrite(output_path, resized_image)

        return True        
    except Exception as e:
        print(f"Error processing {os.path.basename(dicom_path)}: {e}")
        return False

def main():
    print("===== Starting DICOM to PNG Conversion Process for VinDr-CXR =====")

    for set_name, csv_filename in SETS_TO_CONVERT.items():
        print(f"\n--- Processing '{set_name}' set ---")
        
        # Tạo đường dẫn
        source_image_dir = os.path.join(VINDR_ROOT_PATH, set_name)
        source_csv_path = os.path.join(VINDR_ROOT_PATH, csv_filename)
        
        output_image_dir = os.path.join(OUTPUT_ROOT_PATH, set_name)
        os.makedirs(output_image_dir, exist_ok=True)
        
        if not os.path.exists(source_csv_path) or not os.path.exists(source_image_dir):
            print(f"Warning: Path for '{set_name}' set not found. Skipping.")
            continue
            
        # Đọc file CSV để lấy danh sách image_id
        try:
            df = pd.read_csv(source_csv_path)
            image_ids = df['image_id'].unique()
            print(f"Found {len(image_ids)} unique images to convert for the '{set_name}' set.")
        except Exception as e:
            print(f"Error reading CSV file {source_csv_path}: {e}. Skipping this set.")
            continue

        # Lặp qua từng ảnh để chuyển đổi
        success_count = 0
        fail_count = 0
        
        for image_id in tqdm(image_ids, desc=f"Converting {set_name} images"):
            dicom_path = os.path.join(source_image_dir, f"{image_id}.dicom")
            # Lưu file với định dạng PNG
            output_path = os.path.join(output_image_dir, f"{image_id}.png")
            
            # Chỉ chuyển đổi nếu file chưa tồn tại để có thể chạy lại
            if not os.path.exists(output_path):
                if process_dicom_to_png(dicom_path, output_path, target_size=TARGET_SIZE):
                    success_count += 1
                else:
                    fail_count += 1
            else:
                success_count += 1 # Coi như thành công nếu đã tồn tại

        print(f"Conversion for '{set_name}' set completed. Success: {success_count}, Failed: {fail_count}")

    print("\n===== All conversion tasks completed! =====")
    print(f"Converted images are saved in: {OUTPUT_ROOT_PATH}")


if __name__ == "__main__":
    main()