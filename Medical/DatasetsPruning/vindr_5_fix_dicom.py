import pydicom
import os
import warnings

# Directory containing DICOM files
dicom_dir = "/Users/admin/Working/Data/MixData/vinbigdata_structured/images/"

# Suppress invalid VR UI warnings (optional, see notes below)
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom.valuerep")

# Counter for processed and skipped files
processed_files = 0
skipped_files = 0
error_files = []

# Iterate through all files in the directory
for filename in os.listdir(dicom_dir):
    # Skip hidden files (e.g., starting with '._')
    if filename.startswith("._"):
        print(f"Skipping hidden file: {filename}")
        skipped_files += 1
        continue
    
    if filename.endswith(".dicom"):
        file_path = os.path.join(dicom_dir, filename)
        try:
            # Read DICOM file, force reading if header is missing
            ds = pydicom.dcmread(file_path, force=True)
            
            # Check if the file is a valid DICOM file
            if not hasattr(ds, 'file_meta') or ds.file_meta is None:
                print(f"Skipping {filename}: Invalid DICOM file (no file meta information).")
                skipped_files += 1
                continue
                
            # Check and update 'Bits Stored' if necessary
            if hasattr(ds, 'BitsStored') and ds.BitsStored != 16:
                ds.BitsStored = 16
                ds.BitsAllocated = 16
                ds.HighBit = 15
                # Save the modified file
                ds.save_as(file_path)
                print(f"Updated: {filename}")
                processed_files += 1
            else:
                print(f"No update needed: {filename}")
                skipped_files += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            error_files.append(filename)
            continue

# Summary
print("\n--- Processing Summary ---")
print(f"Files processed and updated: {processed_files}")
print(f"Files skipped (no update needed or invalid): {skipped_files}")
print(f"Files with errors: {len(error_files)}")
if error_files:
    print("Files with errors:", ", ".join(error_files))