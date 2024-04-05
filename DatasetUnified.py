import shutil
import random
import pandas as pd
from pathlib import Path
import numpy as np
import yaml
from PIL import Image

# Define the paths to the dataset folders
military_source_dir = Path('Military/dataset')
fvgc_source_dir = Path('FVGC/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data')
combined_target_dir = Path('UnifiedDataset')

# Create target directories for the combined dataset
splits = ['train', 'valid', 'test']
for split in splits:
    for folder in ['images', 'labels']:
        (combined_target_dir / split / folder).mkdir(parents=True, exist_ok=True)

# Define class mappings for both datasets
military_classes = ['A10', 'A400M', 'AG600', 'AV8B', 'B1', 'B2', 'B52', 'Be200', 'C2', 'C17', 'C5', 'E2', 'E7', 'EF2000',
                    'F117', 'F14', 'F15', 'F18', 'F22', 'F35', 'F4', 'JAS39', 'MQ9', 'Mig31', 'Mirage2000', 'P3', 'RQ4', 'Rafale',
                    'SR71', 'Su34', 'Su57', 'Tu160', 'Tu95', 'Tornado', 'U2', 'US2', 'V22', 'XB70', 'YF23', 'Vulcan', 'J20', "F16", "C130", "KC135", "Su25", "J10"]

fvgc_classes = [
    'A300', 'A310', 'A320', 'A330', 'A340', 'A380', 'ATR-42', 'ATR-72', 'An-12', 'BAE 146', 'BAE-125', 'Beechcraft 1900',
    'Boeing 707', 'Boeing 717', 'Boeing 727', 'Boeing 737', 'Boeing 747', 'Boeing 757', 'Boeing 767', 'Boeing 777',
    'C-130', 'C-47', 'CRJ-200', 'CRJ-700', 'Cessna 172', 'Cessna 208', 'Cessna Citation', 'Challenger 600', 'DC-10',
    'DC-3', 'DC-6', 'DC-8', 'DC-9', 'DH-82', 'DHC-1', 'DHC-6', 'DR-400', 'Dash 8', 'Dornier 328', 'EMB-120',
    'Embraer E-Jet', 'Embraer ERJ 145', 'Embraer Legacy 600', 'Eurofighter Typhoon', 'F-16', 'F/A-18', 'Falcon 2000',
    'Falcon 900', 'Fokker 100', 'Fokker 50', 'Fokker 70', 'Global Express', 'Gulfstream', 'Hawk T1', 'Il-76', 'King Air',
    'L-1011', 'MD-11', 'MD-80', 'MD-90', 'Metroliner', 'PA-28', 'SR-20', 'Saab 2000', 'Saab 340', 'Spitfire', 'Tornado',
    'Tu-134', 'Tu-154', 'Yak-42', 'J10', 'Su25','KC135'
]

class_dict = {0: 'A10', 1: 'A400M', 2: 'AG600', 3: 'AV8B', 4: 'B1', 5: 'B2', 6: 'B52', 7: 'Be200', 8: 'C2',
                9: 'C17', 10: 'C5', 11: 'E2', 12: 'E7', 13: 'EF2000', 14: 'F117', 15: 'F14', 16: 'F15', 17: 'F18',
                18: 'F22', 19: 'F35', 20: 'F4', 21: 'JAS39', 22: 'MQ9', 23: 'Mig31', 24: 'Mirage2000', 25: 'P3',
                26: 'RQ4', 27: 'Rafale', 28: 'SR71', 29: 'Su34', 30: 'Su57', 31: 'Tu160', 32: 'Tu95', 33: 'Tornado',
                34: 'U2', 35: 'US2', 36: 'V22', 37: 'XB70', 38: 'YF23', 39: 'Vulcan', 40: 'J20', 41: 'F16',
                42: 'C130', 43: 'KC135', 44: 'Su25', 45: 'J10', 46: 'A300', 47: 'A310', 48: 'A320', 49: 'A330',
                50: 'A340', 51: 'A380', 52: 'ATR-42', 53: 'ATR-72', 54: 'An-12', 55: 'BAE 146', 56: 'BAE-125',
                57: 'Beechcraft 1900', 58: 'Boeing 707', 59: 'Boeing 717', 60: 'Boeing 727', 61: 'Boeing 737',
                62: 'Boeing 747', 63: 'Boeing 757', 64: 'Boeing 767', 65: 'Boeing 777', 66: 'C-47', 67: 'CRJ-200',
                68: 'CRJ-700', 69: 'Cessna 172', 70: 'Cessna 208', 71: 'Cessna Citation', 72: 'Challenger 600',
                73: 'DC-10', 74: 'DC-3', 75: 'DC-6', 76: 'DC-8', 77: 'DC-9', 78: 'DH-82', 79: 'DHC-1', 80: 'DHC-6',
                81: 'DR-400', 82: 'Dash 8', 83: 'Dornier 328', 84: 'EMB-120', 85: 'Embraer E-Jet', 86: 'Embraer ERJ 145',
                87: 'Embraer Legacy 600', 88: 'Eurofighter Typhoon', 89: 'Falcon 2000', 90: 'Falcon 900', 91: 'Fokker 100',
                92: 'Fokker 50', 93: 'Fokker 70', 94: 'Global Express', 95: 'Gulfstream', 96: 'Hawk T1', 97: 'Il-76',
                98: 'King Air', 99: 'L-1011', 100: 'MD-11', 101: 'MD-80', 102: 'MD-90', 103: 'Metroliner', 104: 'PA-28',
                105: 'SR-20', 106: 'Saab 2000', 107: 'Saab 340', 108: 'Spitfire', 109: 'Tu-134', 110: 'Tu-154', 111: 'Yak-42'}

inverted_class_dict = {v: k for k, v in class_dict.items()}

# Handle overlapping classes by renaming them in the FVGC dataset
class_remap = {"F-16": "F16", "C-130": "C130", "F/A-18": "F18"}

# Combine both class lists into a unified list without duplicates
combined_classes = list(set(military_classes + [class_remap.get(c, c) for c in fvgc_classes]))

# Function to get the class index from the combined class dictionary
def get_class_index(class_name):
    remapped_name = class_remap.get(class_name, class_name)
    if remapped_name not in class_dict.values():
        raise ValueError(f"Class name '{remapped_name}' is not in the class dictionary.")
    return list(class_dict.keys())[list(class_dict.values()).index(remapped_name)]

# Function to read image class files and return a dictionary mapping image names to class labels
def read_image_classes(file_path):
    image_classes = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if len(parts) == 2:
                image_name, image_class = parts
                image_classes[image_name] = class_remap.get(image_class, image_class)  # Remap if needed
    return image_classes


# Function to process FVGC dataset and create YOLO labels
def process_fvgc_dataset(image_lists, box_data, class_data, target_dir):
    for split, image_list in image_lists.items():
        labels_dir = target_dir / split / 'labels'
        images_dir = target_dir / split / 'images'
        for image_name in image_list:
            # Get bounding box and class name, remap class name if necessary
            xmin, ymin, xmax, ymax = box_data.get(image_name, (0, 0, 0, 0))
            class_name = class_data.get(image_name, "Unknown")
            class_index = get_class_index(class_name)

            # Load image to get its dimensions
            image_file_path = fvgc_source_dir / 'images' / f"{image_name}.jpg"
            with Image.open(image_file_path) as img:
                img_width, img_height = img.size

            # Normalize coordinates
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            bbox_width = (xmax - xmin) / img_width
            bbox_height = (ymax - ymin) / img_height

            # Create YOLO formatted label file
            label_content = f"{class_index} {x_center} {y_center} {bbox_width} {bbox_height}\n"
            label_file_path = labels_dir / f"{image_name}.txt"
            with open(label_file_path, 'w') as label_file:
                label_file.write(label_content)

            # Copy image file to the target directory
            target_image_path = images_dir / f"{image_name}.jpg"
            if image_file_path.exists():
                shutil.copy(image_file_path, target_image_path)

def read_image_boxes(file_path):
    image_boxes = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if len(parts) == 5:
                image_name, xmin, ymin, xmax, ymax = parts
                image_boxes[image_name] = (int(xmin), int(ymin), int(xmax), int(ymax))
    return image_boxes


# Read class and box data from FVGC dataset
fvgc_test_classes = read_image_classes(fvgc_source_dir / 'images_family_test.txt')
fvgc_train_classes = read_image_classes(fvgc_source_dir / 'images_family_train.txt')
fvgc_val_classes = read_image_classes(fvgc_source_dir / 'images_family_val.txt')
fvgc_box_data = read_image_boxes(fvgc_source_dir / 'images_box.txt')

# Organize FVGC image lists by split
fvgc_image_lists = {
    'test': list(fvgc_test_classes.keys()),
    'train': list(fvgc_train_classes.keys()),
    'valid': list(fvgc_val_classes.keys())
}

# Process FVGC dataset and create labels
process_fvgc_dataset(fvgc_image_lists, fvgc_box_data, {**fvgc_test_classes, **fvgc_train_classes, **fvgc_val_classes}, combined_target_dir)


print(f"Combined dataset created at {combined_target_dir}")

def process_dataset(file_indices, file_type):
    for idx in file_indices:
        # Read CSV and extract bounding box information
        csv_file = csv_files[idx]
        df = pd.read_csv(csv_file)

        # Prepare YOLO formatted labels
        label_content = ''
        for _, row in df.iterrows():
            class_name = row['class']
            class_id = inverted_class_dict.get(class_name)
            if class_id is None:
                print(f"Class name '{class_name}' not found in class dictionary. Skipping entry.")
                continue

            # Assuming the CSV has 'xmin', 'ymin', 'xmax', 'ymax' columns
            x_center = (row['xmin'] + row['xmax']) / 2
            y_center = (row['ymin'] + row['ymax']) / 2
            width = row['xmax'] - row['xmin']
            height = row['ymax'] - row['ymin']

            # Assuming the CSV has 'width' and 'height' columns for the image dimensions
            label_content += f"{class_id} {x_center / row['width']} {y_center / row['height']} {width / row['width']} {height / row['height']}\n"

        # Write YOLO formatted label file
        label_file = combined_target_dir / file_type / 'labels' / csv_file.with_suffix('.txt').name
        with open(label_file, 'w') as file:
            file.write(label_content)

        # Copy the corresponding image
        img_file = jpg_files[idx]
        target_img_path = combined_target_dir / file_type / 'images' / img_file.name
        shutil.copy(img_file, target_img_path)


# Get all CSV and JPG filenames
csv_files = sorted(military_source_dir.glob('*.csv'))
jpg_files = sorted(military_source_dir.glob('*.jpg'))

# Randomly split files into train, valid, and test sets
np.random.seed(42)
total_files = len(csv_files)
train_ratio, valid_ratio = 0.8, 0.1  # Remaining for testing
train_indices = np.random.choice(total_files, int(total_files * train_ratio), replace=False)
valid_indices = np.random.choice(list(set(range(total_files)) - set(train_indices)), int(total_files * valid_ratio),
                                 replace=False)
test_indices = list(set(range(total_files)) - set(train_indices) - set(valid_indices))

# Process files for each dataset split
for file_type, indices in [('train', train_indices), ('valid', valid_indices), ('test', test_indices)]:
    process_dataset(indices, file_type)

print("Dataset processing complete.")


# Convert the combined list to a NumPy array and then to a list of Python strings
classes = [str(cls) for cls in np.array(combined_classes)]
names = {idx: cls for idx, cls in enumerate(classes)}

# YOLO configuration dictionary
yolo_config = {
    'path': '/content/drive/Othercomputers/My MacBook Pro/UnifiedDataset',
    'train': 'images/train',
    'val': 'images/valid',
    'test': 'images/test',
    #'nc': len(classes),
    #'names': classes,
    #'names': {idx: cls for idx, cls in enumerate(classes)}
    #'names': class_dict
    'names': yaml.dump(class_dict, default_flow_style=False)

}

# Specify the directory and file name for the YAML file
yaml_path = 'UnifiedDataset/data.yaml'

# Save the YAML file
with open(yaml_path, 'w') as yaml_file:
    yaml.dump(yolo_config, yaml_file, default_flow_style=None)
    yaml.dump(class_dict, yaml_file, default_flow_style=False)

print(f"YAML file saved to: {yaml_path}")
