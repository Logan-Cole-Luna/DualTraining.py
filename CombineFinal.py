from pathlib import Path
import os
import shutil
from glob import glob

# Paths to the source datasets
source1 = Path(r"C:\Users\Logan\Downloads\militaryAircraft.v9i.yolov8")
source2 = "UnifiedDataset"

# Destination directory (assuming you're combining into the UnifiedDataset structure)
dest = "CombinedDataset"


# Function to copy and possibly rename files from source to destination
def copy_and_maybe_rename(source_dir, dest_dir):
    for filepath in glob(os.path.join(source_dir, '*')):
        filename = os.path.basename(filepath)
        dest_path = os.path.join(dest_dir, filename)

        # Check if the file already exists in the destination directory
        if os.path.exists(dest_path):
            # Create a new filename to avoid overwriting
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_copy{ext}"
            dest_path = os.path.join(dest_dir, new_filename)

        shutil.copy(filepath, dest_path)


# Categories and types of data
categories = ['train', 'test']
data_types = ['labels', 'images']

# Combine the datasets
for category in categories:
    for data_type in data_types:
        # Define source and destination directories
        source_dir1 = os.path.join(source1, category, data_type)
        source_dir2 = os.path.join(source2, category, data_type)
        dest_dir = os.path.join(dest, category, data_type)

        # Ensure destination directory exists
        os.makedirs(dest_dir, exist_ok=True)

        # Copy from source1 and source2 to destination
        copy_and_maybe_rename(source_dir1, dest_dir)
        copy_and_maybe_rename(source_dir2, dest_dir)

# Print the size of the combined dataset
for category in categories:
    for data_type in data_types:
        dest_dir = os.path.join(dest, category, data_type)
        file_count = len(glob(os.path.join(dest_dir, '*')))
        print(f"{category}/{data_type}: {file_count} files")
