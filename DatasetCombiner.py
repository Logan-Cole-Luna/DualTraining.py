import os
import shutil
from glob import glob
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

def find_image_in_file(file_path, image_number):
    try:
        with open(file_path, 'r') as file:
            data = file.read()
            print("Found")
            return image_number in data
    except Exception as e:
        print("Error")
        return str(e)

# Define the file path and the image number to search for
file_path = 'FVGC/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images_family_test.txt'


# Function to read the class names associated with each image from the provided files
def read_class_names(file_path):
    class_names = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                image_name, class_name = parts
                class_names[image_name] = class_name
    return class_names

def read_bounding_box_data(file_path):
    bounding_box_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                image_id, xmin, ymin, xmax, ymax = parts
                bounding_box_data[image_id] = (int(xmin), int(ymin), int(xmax), int(ymax))
    return bounding_box_data


def read_image_list(file_path):
    with open(file_path, 'r') as file:
        return [line.strip().split()[0] for line in file.readlines()]



# Define source and target directories
source_dir = 'Military/dataset'
target_dir = 'MilitaryData2/'

fvgc_source_dir = 'FVGC/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data'

# Create target directories
splits = ['train', 'valid', 'test']
for split in splits:
    for folder in ['images', 'labels']:
        dir_path = os.path.join(target_dir, split, folder)
        os.makedirs(dir_path, exist_ok=True)


# Function to copy image files based on class names list
def copy_files(class_names, split, source_dir, target_dir):
    for image_name in class_names.keys():
        source_image_file = os.path.join(source_dir, f'images',f'{image_name}.jpg')  # Update if FVGC has a different naming convention
        target_image_dir = os.path.join(target_dir, split, 'images', f'{image_name}.jpg')
        if os.path.exists(source_image_file):
            shutil.copy(source_image_file, target_image_dir)
        else:
            print(f"Image file does not exist: {source_image_file}")

# Read class names and bounding box data
test_classes = read_class_names('FVGC/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images_family_test.txt')
train_classes = read_class_names('FVGC/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images_family_train.txt')
val_classes = read_class_names('FVGC/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images_family_val.txt')

box_data_path = 'FVGC/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images_box.txt'
bounding_box_data = read_bounding_box_data(box_data_path)

# Copy files for each dataset split using the FVGC source directory
copy_files(train_classes, 'train', fvgc_source_dir, target_dir)
copy_files(val_classes, 'valid', fvgc_source_dir, target_dir)
copy_files(test_classes, 'test', fvgc_source_dir, target_dir)
print("Files copied successfully based on train, valid, and test specifications.")

'''''''''



# Function to create CSV files with bounding box data and correct class names
def create_csv_files_with_bbox_and_class(image_list, split, box_data, class_names, target_dir):
    for image_name in image_list:
        if image_name in box_data:
            xmin, ymin, xmax, ymax = box_data[image_name]
            width, height = xmax - xmin, ymax - ymin
            class_name = combined_classes.get(image_name)
            if class_name is None:  # No class name found for this image
                is_image_present = find_image_in_file(file_path, image_name)
                print(f"Warning: Class name for image '{image_name}' not found, defaulting to 'Unknown'")
                class_name = "Unknown"
            df = pd.DataFrame({
                'filename': [image_name],
                'width': [width],
                'height': [height],
                'class': [class_name],
                'xmin': [xmin],
                'ymin': [ymin],
                'xmax': [xmax],
                'ymax': [ymax]
            })
            csv_path = Path(target_dir) / split / 'labels' / f'{image_name}.csv'
            df.to_csv(csv_path, index=False)

test_images_path = 'FVGC/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images_family_test.txt'
train_images_path = 'FVGC/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images_family_train.txt'
val_images_path = 'FVGC/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images_family_val.txt'

# Process FVGC dataset with updated class names
split_class_mapping = {'test': test_classes, 'train': train_classes, 'valid': val_classes}
for split, images_path in [('test', test_images_path), ('train', train_images_path), ('valid', val_images_path)]:
    images = read_image_list(images_path)
    print(split)
    create_csv_files_with_bbox_and_class(images, split, bounding_box_data, split_class_mapping[split], target_dir)
    for image_name in images:
        source_image_path = Path('FVGC/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images') / f'{image_name}.jpg'
        target_image_path = Path(target_dir) / split / 'images' / f'{image_name}.jpg'

        # Check if the source image exists
        if source_image_path.exists():
            shutil.copy(source_image_path, target_image_path)
        else:
            print(f"Source file does not exist: {source_image_path}")

print(f"FVGC Dataset formatted and saved in: {target_dir}")


'''''''''


# Function to update class names in the CSV files
def update_class_names_in_csv(file_path, old_class_name, new_class_name):
    df = pd.read_csv(file_path)
    df['class'] = df['class'].replace(old_class_name, new_class_name)
    df.to_csv(file_path, index=False)


# Get all CSV and JPG filenames
csv_files = sorted(glob(os.path.join(source_dir, '*.csv')))
jpg_files = sorted(glob(os.path.join(source_dir, '*.jpg')))
print(f"Found {len(csv_files)} CSV files and {len(jpg_files)} JPG files in {source_dir}")

# Randomly split files into train, valid, and test sets
np.random.seed(42)
total_files = len(csv_files)
train_ratio, valid_ratio = 0.8, 0.1  # Remaining 10% for testing
train_indices = np.random.choice(total_files, int(total_files * train_ratio), replace=False)
valid_indices = np.random.choice(list(set(range(total_files)) - set(train_indices)), int(total_files * valid_ratio), replace=False)
test_indices = list(set(range(total_files)) - set(train_indices) - set(valid_indices))

# Function to copy files based on indices
def copy_filess(file_indices, file_type):
    for idx in file_indices:
        shutil.copy(csv_files[idx], os.path.join(target_dir, file_type, 'labels', os.path.basename(csv_files[idx])))
        shutil.copy(jpg_files[idx], os.path.join(target_dir, file_type, 'images', os.path.basename(jpg_files[idx])))

# Copy files to their corresponding directories
for file_type, indices in [('train', train_indices), ('valid', valid_indices), ('test', test_indices)]:
    copy_filess(indices, file_type)

print("Files copied successfully.")













'''''''''
# Update class names in CSV files in the target directory
for split in splits:
    labels_dir = os.path.join(target_dir, split, 'labels')
    for csv_file in glob(os.path.join(labels_dir, '*.csv')):
        update_class_names_in_csv(csv_file, 'F16', 'F-16')
        update_class_names_in_csv(csv_file, 'C130', 'C-130')
'''''''''

# Original classes array
original_classes = ['A10', 'A400M', 'AG600', 'AV8B', 'B1', 'B2', 'B52', 'Be200', 'C2', 'C17', 'C5', 'E2', 'E7', 'EF2000',
                    'F117', 'F14', 'F15', 'F18', 'F22', 'F35', 'F4', 'JAS39', 'MQ9', 'Mig31', 'Mirage2000', 'P3', 'RQ4', 'Rafale',
                    'SR71', 'Su34', 'Su57', 'Tu160', 'Tu95', 'Tornado', 'U2', 'US2', 'V22', 'XB70', 'YF23', 'Vulcan', 'J20']

'''''''''
# New family classes from the provided text
family_classes = [
    'A300', 'A310', 'A320', 'A330', 'A340', 'A380', 'ATR-42', 'ATR-72', 'An-12', 'BAE 146', 'BAE-125', 'Beechcraft 1900',
    'Boeing 707', 'Boeing 717', 'Boeing 727', 'Boeing 737', 'Boeing 747', 'Boeing 757', 'Boeing 767', 'Boeing 777',
    'C-130', 'C-47', 'CRJ-200', 'CRJ-700', 'Cessna 172', 'Cessna 208', 'Cessna Citation', 'Challenger 600', 'DC-10',
    'DC-3', 'DC-6', 'DC-8', 'DC-9', 'DH-82', 'DHC-1', 'DHC-6', 'DR-400', 'Dash 8', 'Dornier 328', 'EMB-120',
    'Embraer E-Jet', 'Embraer ERJ 145', 'Embraer Legacy 600', 'Eurofighter Typhoon', 'F-16', 'F/A-18', 'Falcon 2000',
    'Falcon 900', 'Fokker 100', 'Fokker 50', 'Fokker 70', 'Global Express', 'Gulfstream', 'Hawk T1', 'Il-76', 'King Air',
    'L-1011', 'MD-11', 'MD-80', 'MD-90', 'Metroliner', 'PA-28', 'SR-20', 'Saab 2000', 'Saab 340', 'Spitfire', 'Tornado',
    'Tu-134', 'Tu-154', 'Yak-42', 'J10', 'Su25','KC135'
]
'''''''''

# Combine the original classes with the family classes
#combined_classes = set(original_classes) | set(family_classes)

# Convert the combined set to a NumPy array
classes = np.array(list(original_classes))

# Convert numpy array elements to plain Python strings
classes = [str(cls) for cls in classes]

# Print the combined classes
print(classes)
print(len(classes))

# Creating the YAML file for YOLOv8
yolo_config = {
    'train': '/content/drive/MyDrive/MilitaryData2/train/images',
    'val': '/content/drive/MyDrive/MilitaryData2/valid/images',
    'test': '/content/drive/MyDrive/MilitaryData2/test/images',
    'names': {idx: cls for idx, cls in enumerate(classes)}
}

# Specify a writable directory and file name for the YAML file
yaml_path = 'MilitaryData2/mad.yaml'

# Save the YAML file
with open(yaml_path, 'w') as yaml_file:
    yaml.dump(yolo_config, yaml_file, default_flow_style=False)

print(f"YAML file saved to: {yaml_path}")
