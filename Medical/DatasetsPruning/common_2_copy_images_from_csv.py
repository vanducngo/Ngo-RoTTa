import pandas as pd
import shutil
import os

# File paths
# train_output_file = '/Users/admin/Working/Data/MixData/nih_14_structured/validate.csv'
# source_image_dir = '/Users/admin/Working/Data/MixData/nih_14_structured/images'
# destination_image_dir = '/Users/admin/Working/Data/MixData/nih_14_structured_filtered/images'

train_output_file = '/Users/admin/Working/Data/MixData/vinbigdata_structured/validate.csv'
source_image_dir = '/Users/admin/Working/Data/MixData/vinbigdata_structured/images'
destination_image_dir = '/Users/admin/Working/Data/MixData/vinbigdata_structured_filtered/images'


def copy_filtered_images(csv_file, source_dir, dest_dir):
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Read the filtered CSV file
    df = pd.read_csv(csv_file)

    # Get list of image IDs
    image_ids = df['image_id'].tolist()

    # Copy images
    copied_count = 0
    for image_id in image_ids:
        image_name = f"{image_id}"
        source_path = os.path.join(source_dir, image_name)
        dest_path = os.path.join(dest_dir, image_name)

        # Check if source image exists
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            copied_count += 1
            print(f"Copied image: {image_id}")
        else:
            print(f"Image not found: {source_path}")

    print(f"Copied {copied_count} images to {dest_dir}")

if __name__ == "__main__":
    copy_filtered_images(train_output_file, source_image_dir, destination_image_dir)