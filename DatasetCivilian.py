import shutil
import random
import pandas as pd
from pathlib import Path
import numpy as np
import yaml
from PIL import Image

# Define the paths to the dataset folders
#military_source_dir = Path('Military/dataset')
fvgc_source_dir = Path(r"B:\Datasets\archive\fgvc-aircraft-2013b\fgvc-aircraft-2013b\data")
combined_target_dir = Path('CivilianDataset')

# Create target directories for the combined dataset
splits = ['train', 'valid', 'test']
for split in splits:
    for folder in ['images', 'labels']:
        (combined_target_dir / split / folder).mkdir(parents=True, exist_ok=True)

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

class_dict = {
 0: 'A300',
 1: 'A310',
 2: 'A320',
 3: 'A330',
 4: 'A340',
 5: 'A380',
 6: 'ATR-42',
 7: 'ATR-72',
 8: 'An-12',
 9: 'BAE 146',
 10: 'BAE-125',
 11: 'Beechcraft 1900',
 12: 'Boeing 707',
 13: 'Boeing 717',
 14: 'Boeing 727',
 15: 'Boeing 737',
 16: 'Boeing 747',
 17: 'Boeing 757',
 18: 'Boeing 767',
 19: 'Boeing 777',
 20: 'C-130',
 21: 'C-47',
 22: 'CRJ-200',
 23: 'CRJ-700',
 24: 'Cessna 172',
 25: 'Cessna 208',
 26: 'Cessna Citation',
 27: 'Challenger 600',
 28: 'DC-10',
 29: 'DC-3',
 30: 'DC-6',
 31: 'DC-8',
 32: 'DC-9',
 33: 'DH-82',
 34: 'DHC-1',
 35: 'DHC-6',
 36: 'DR-400',
 37: 'Dash 8',
 38: 'Dornier 328',
 39: 'EMB-120',
 40: 'Embraer E-Jet',
 41: 'Embraer ERJ 145',
 42: 'Embraer Legacy 600',
 43: 'Eurofighter Typhoon',
 44: 'F-16',
 45: 'F/A-18',
 46: 'Falcon 2000',
 47: 'Falcon 900',
 48: 'Fokker 100',
 49: 'Fokker 50',
 50: 'Fokker 70',
 51: 'Global Express',
 52: 'Gulfstream',
 53: 'Hawk T1',
 54: 'Il-76',
 55: 'King Air',
 56: 'L-1011',
 57: 'MD-11',
 58: 'MD-80',
 59: 'MD-90',
 60: 'Metroliner',
 61: 'PA-28',
 62: 'SR-20',
 63: 'Saab 2000',
 64: 'Saab 340',
 65: 'Spitfire',
 66: 'Tornado',
 67: 'Tu-134',
 68: 'Tu-154',
 69: 'Yak-42',
 70: 'J10',
 71: 'Su25',
 72: 'KC135'
}

inverted_class_dict = {v: k for k, v in class_dict.items()}

# Handle overlapping classes by renaming them in the FVGC dataset
class_remap = {"F-16": "F-16", "C-130": "C-130", "F/A-18": "F/A-18"}

# Function to get the class index from the combined class dictionary
def get_class_index(class_name):
    '''    remapped_name = class_remap.get(class_name, class_name)
    if remapped_name not in class_dict.values():
        raise ValueError(f"Class name '{remapped_name}' is not in the class dictionary.")'''
    return list(class_dict.keys())[list(class_dict.values()).index(class_name)]

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


'''''''''
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

'''''''''
print("Dataset processing complete.")


# Convert the combined list to a NumPy array and then to a list of Python strings
classes = [str(cls) for cls in np.array(fvgc_classes)]
names = {idx: cls for idx, cls in enumerate(classes)}

# YOLO configuration dictionary
yolo_config = {
    'path': 'B:\GitHub\DualTraining.py\CivilianDataset',
    'train': 'images/train',
    #'val': 'images/valid',
    'test': 'images/test',
    #'nc': len(classes),
    #'names': classes,
    #'names': {idx: cls for idx, cls in enumerate(classes)}
    #'names': class_dict
    'names': yaml.dump(class_dict, default_flow_style=False)

}

# Specify the directory and file name for the YAML file
yaml_path = 'CivilianDataset/data.yaml'

# Save the YAML file
with open(yaml_path, 'w') as yaml_file:
    yaml.dump(yolo_config, yaml_file, default_flow_style=None)
    yaml.dump(class_dict, yaml_file, default_flow_style=False)

print(f"YAML file saved to: {yaml_path}")
