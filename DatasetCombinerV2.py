import pandas as pd
from pathlib import Path
import os

# Function to read image class files and return a dictionary mapping image names to class labels
def read_image_classes(file_path):
    image_classes = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if len(parts) == 2:
                image_name, image_class = parts
                image_classes[image_name] = image_class
    return image_classes

# Function to read the bounding box file and return a dictionary mapping image names to bounding boxes
def read_image_boxes(file_path):
    image_boxes = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if len(parts) == 5:
                image_name, xmin, ymin, xmax, ymax = parts
                image_boxes[image_name] = (int(xmin), int(ymin), int(xmax), int(ymax))
    return image_boxes

# Function to find if an image is present in the dataset (Assuming we have a function to do this)
def find_image_in_file(file_path, image_name):
    # This is a placeholder function. The implementation would depend on the dataset structure.
    # For now, we assume that the image is found.
    return True

# Main function to create CSV files with bounding box and class
def create_csv_files_with_bbox_and_class(image_lists, box_data, class_data, target_dir):
    Path(target_dir).mkdir(parents=True, exist_ok=True)  # Ensure the target directory exists

    for split, image_list in image_lists.items():
        labels_dir = Path(target_dir) / split / 'labels'
        labels_dir.mkdir(parents=True, exist_ok=True)  # Ensure the labels directory exists for each split

        for image_name in image_list:
            if image_name in box_data:
                xmin, ymin, xmax, ymax = box_data[image_name]
                width, height = xmax - xmin, ymax - ymin
                class_name = class_data.get(image_name, "Unknown")

                if class_name == "Unknown":  # No class name found for this image
                    # Implement the find_image_in_file function according to your dataset
                    is_image_present = find_image_in_file("FVGC/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images", image_name)
                    print("Error")
                    if not is_image_present:
                        print(f"Warning: Image '{image_name}' not found in dataset.")
                        continue

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
                csv_path = labels_dir / f'{image_name}.csv'
                df.to_csv(csv_path, index=False)


# Read class names and bounding box data
test_classes = read_image_classes('FVGC/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images_family_test.txt')
train_classes = read_image_classes('FVGC/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images_family_train.txt')
val_classes = read_image_classes('FVGC/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images_family_val.txt')

box_data_path = 'FVGC/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images_box.txt'
box_data = read_image_boxes(box_data_path)

# Combine all class data
all_classes = {**test_classes, **train_classes, **val_classes}

# Organize image lists by split
image_lists = {
    'test': list(test_classes.keys()),
    'train': list(train_classes.keys()),
    'valid': list(val_classes.keys())
}

# Set the target directory for CSV files
target_dir = 'CombinedDataset/'

# Call the function to create CSV files
create_csv_files_with_bbox_and_class(image_lists, box_data, all_classes, target_dir)

# Print out the path to the directory containing the CSV files
print(f"CSV files have been created in the directory: {target_dir}")
