import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import yaml

# Set the directory where your Military dataset CSV and JPG files are
source_dir = Path('Military/dataset')

# Set the directory where you want to save the organized dataset
target_dir = Path('MilitaryData2')

# Create the necessary directories
splits = ['train', 'valid', 'test']
for split in splits:
    for folder in ['images', 'labels']:
        (target_dir / split / folder).mkdir(parents=True, exist_ok=True)

# Dictionary to map class names to YOLO class indices
class_dict = {
    "A10": 0, "A400M": 1, "AG600": 2, "AV8B": 3, "B1": 4, "B2": 5, "B52": 6, "Be200": 7, "C2": 8, "C17": 9,
    "C5": 10, "E2": 11, "E7": 12, "EF2000": 13, "F117": 14, "F14": 15, "F15": 16, "F18": 17, "F22": 18, "F35": 19,
    "F4": 20, "JAS39": 21, "MQ9": 22, "Mig31": 23, "Mirage2000": 24, "P3": 25, "RQ4": 26, "Rafale": 27, "SR71": 28,
    "Su34": 29, "Su57": 30, "Tu160": 31, "Tu95": 32, "Tornado": 33, "U2": 34, "US2": 35, "V22": 36, "XB70": 37,
    "YF23": 38, "Vulcan": 39, "J20": 40, "F16": 41, "C130": 42, "KC135": 43, "Su25": 44, "J10": 45
}

# Function to process CSV files and organize dataset
def process_dataset(file_indices, file_type):
    for idx in file_indices:
        # Read CSV and extract bounding box information
        csv_file = csv_files[idx]
        df = pd.read_csv(csv_file)

        # Prepare YOLO formatted labels
        label_content = ''
        for _, row in df.iterrows():
            class_name = row['class']
            class_id = class_dict.get(class_name)
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
        label_file = target_dir / file_type / 'labels' / csv_file.with_suffix('.txt').name
        with open(label_file, 'w') as file:
            file.write(label_content)

        # Copy the corresponding image
        img_file = jpg_files[idx]
        target_img_path = target_dir / file_type / 'images' / img_file.name
        shutil.copy(img_file, target_img_path)


# Get all CSV and JPG filenames
csv_files = sorted(source_dir.glob('*.csv'))
jpg_files = sorted(source_dir.glob('*.jpg'))

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

original_classes = ['A10', 'A400M', 'AG600', 'AV8B', 'B1', 'B2', 'B52', 'Be200', 'C2', 'C17', 'C5', 'E2', 'E7', 'EF2000',
                    'F117', 'F14', 'F15', 'F18', 'F22', 'F35', 'F4', 'JAS39', 'MQ9', 'Mig31', 'Mirage2000', 'P3', 'RQ4', 'Rafale',
                    'SR71', 'Su34', 'Su57', 'Tu160', 'Tu95', 'Tornado', 'U2', 'US2', 'V22', 'XB70', 'YF23', 'Vulcan', 'J20', "F16", "C130", "KC135", "Su25", "J10"]


# Convert the combined set to a NumPy array
classes = np.array(list(original_classes))

# Convert numpy array elements to plain Python strings
classes = [str(cls) for cls in classes]

# Print the combined classes
print(classes)
print(len(classes))


yolo_config = {
    'path': '/content/drive/MyDrive/SmallMilitaryData',
    'train': 'images/train',
    'val': 'images/valid',
    'test': 'images/test',
    'names': {idx: cls for idx, cls in enumerate(classes)}
}

# Specify a writable directory and file name for the YAML file
yaml_path = 'MilitaryData2/mad.yaml'

# Save the YAML file
with open(yaml_path, 'w') as yaml_file:
    yaml.dump(yolo_config, yaml_file, default_flow_style=False)

print(f"YAML file saved to: {yaml_path}")

