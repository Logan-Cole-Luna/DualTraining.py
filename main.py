import numpy as np
import pandas as pd
import glob
import shutil
#import ultralytics
import os
import yaml
#import torch
import zipfile

'''''
pip install numpy

pip install zipfile

git clone https://github.com/WongKinYiu/yolov7.git
pip install -r yolov7/requirements.txt

pip install -r lib/yolov7/requirements.txt

os.chdir('lib/yolov7')

pip install wandb

pip install pycocotools
'''''
#with zipfile.ZipFile('yolov7-main.zip', 'r') as zip_ref:
#  zip_ref.extractall('lib/yolov7')

#
#with zipfile.ZipFile('MilitaryDataset.zip', 'r') as zip_ref:
#  zip_ref.extractall('lib/MilitaryDataset')



#with open('lib/yolov7/detect.py') as f:
#    data_lines = f.read()
#data_lines = data_lines.replace('line_thickness=1', 'line_thickness=4')
#with open('lib/yolov7/detect.py', mode='w') as f:
#    f.write(data_lines)


classes = np.array(['A10', 'A400M', 'AG600', 'AV8B', 'B1', 'B2', 'B52', 'Be200', 'C130', 'C2', 'C17', 'C5', 'E2', 'E7', 'EF2000',
                    'F117', 'F14', 'F15', 'F16', 'F18', 'F22', 'F35', 'F4', 'JAS39', 'MQ9', 'Mig31', 'Mirage2000', 'P3', 'RQ4', 'Rafale',
                    'SR71', 'Su34', 'Su57', 'Tu160', 'Tu95', 'Tornado', 'U2', 'US2', 'V22', 'XB70', 'YF23', 'Vulcan', 'J20'])

print(len(classes))
print("Current Working Directory:", os.getcwd())

#os.chdir('lib/MilitaryDataset/dataset')

directory_to_check = 'lib/MilitaryDataset/dataset'
if os.path.exists(directory_to_check):
    print(f"{directory_to_check} exists.")
else:
    print(f"{directory_to_check} does not exist!")

print(os.getcwd())
csv_paths = sorted(glob.glob('*.csv'))
jpg_paths = sorted(glob.glob('*.jpg'))
print('number of images:', len(csv_paths))

# Create directories
directories_to_create = [
    'lib/MilitaryDataset/data/train/images',
    'lib/MilitaryDataset/data/train/labels',
    'lib/MilitaryDataset/data/valid/images',
    'lib/MilitaryDataset/data/valid/labels',
    'lib/MilitaryDataset/data/test/images',
    'lib/MilitaryDataset/data/test/labels'
]

for directory in directories_to_create:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Successfully created the directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

# Randomly split files into train, valid, and test sets
np.random.seed(42)
total_files = len(csv_paths)
train_ratio, valid_ratio = 0.8, 0.1  # 10% for testing by default
train_files = np.random.choice(total_files, int(total_files * train_ratio), replace=False)
valid_files = np.random.choice(list(set(range(total_files)) - set(train_files)), int(total_files * valid_ratio), replace=False)
test_files = list(set(range(total_files)) - set(train_files) - set(valid_files))
print(f"Number of CSV files: {len(csv_paths)}")
print(csv_paths[:5])  # Print first 5 entries

def save_files(file_indices, file_type):
    for index in file_indices:
        csv_path = csv_paths[index]
        jpg_path = jpg_paths[index]

        # Print paths for debugging
        print(f"Processing CSV: {csv_path}")
        print(f"Processing JPG: {jpg_path}")

        annotations = np.array(pd.read_csv(csv_path))
        if len(annotations) == 0:
            print(f"Warning: No annotations found in {csv_path}")
            continue

        jpg_dest_path = f'lib/MilitaryDataset/data/{file_type}/images/' + os.path.basename(jpg_path)
        txt_dest_path = f'lib/MilitaryDataset/data/{file_type}/labels/' + os.path.basename(csv_path)[:-4] + '.txt'

        # Print destination paths for debugging
        print(f"Copying JPG to: {jpg_dest_path}")
        print(f"Saving annotations to: {txt_dest_path}")

        shutil.move(jpg_path, jpg_dest_path)
        with open(txt_dest_path, mode='w') as f:
            for annotation in annotations:
                width = annotation[1]
                height = annotation[2]
                class_name = annotation[3]
                xmin = annotation[4]
                ymin = annotation[5]
                xmax = annotation[6]
                ymax = annotation[7]
                x_center = 0.5 * (xmin + xmax)
                y_center = 0.5 * (ymin + ymax)
                b_width = xmax - xmin
                b_height = ymax - ymin
                class_num = np.where(classes == class_name)[0][0]
                output_string = '{} {} {} {} {}\n'.format(class_num, x_center / width, y_center / height, b_width / width, b_height / height)
                f.write(output_string)

# Save files to respective directories
save_files(train_files, "train")
save_files(valid_files, "valid")
save_files(test_files, "test")

print('test:', len(glob.glob('lib/MilitaryDataset/data/test/labels/*.txt')))
print('valid:', len(glob.glob('lib/MilitaryDataset/data/valid/labels/*.txt')))
print('train', len(glob.glob('lib/MilitaryDataset/data/train/labels/*.txt')))

# Create mad.yaml
base_path = '/content/MilitaryDataset/dataset'
# Create mad.yaml
classes_string = ', '.join([f'"{c}"' for c in classes])
# Create mad.yaml with absolute paths
yaml_path = os.path.abspath('lib/MilitaryDataset/data/mad.yaml')
train_path = os.path.abspath('lib/MilitaryDataset/data/train/images')
valid_path = os.path.abspath('lib/MilitaryDataset/data/valid/images')
test_path = os.path.abspath('lib/MilitaryDataset/data/test/images')

with open(yaml_path, 'w') as f:
    yaml_content = (
        f'train: {train_path}\n'
        f'val: {valid_path}\n'
        f'test: {test_path}\n'
        f'nc: {len(classes)}\n'
        f'names: [{classes_string}]'
    )
    f.write(yaml_content)

print(f"'mad.yaml' created at {yaml_path}")

os.makedirs('lib/yolov7/checkpoints', exist_ok=True)

hyp = {
    'lr0': 0.0002,  # initial learning rate (SGD=1E-2, Adam=1E-3)
    'lrf': 0.001,  # final OneCycleLR learning rate (lr0 * lrf)
    'momentum': 0.937,  # SGD momentum/Adam beta1
    'weight_decay': 0.0005,  # optimizer weight decay 5e-4
    'warmup_epochs': 2.0,  # warmup epochs (fractions ok)
    'warmup_momentum': 0.8,  # warmup initial momentum
    'warmup_bias_lr': 0.0001,  # warmup initial bias lr
    'box': 0.05,  # box loss gain
    'cls': 0.3,  # cls loss gain
    'cls_pw': 1.0,  # cls BCELoss positive_weight
    'obj': 0.7,  # obj loss gain (scale with pixels)
    'obj_pw': 1.0,  # obj BCELoss positive_weight
    'iou_t': 0.20,  # IoU training threshold
    'anchor_t': 4.0,  # anchor-multiple threshold
    # anchors: 3  # anchors per output layer (0 to ignore)
    'fl_gamma': 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
    'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
    'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
    'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
    'degrees': 0.5,  # image rotation (+/- deg)
    'translate': 0.25,  # image translation (+/- fraction)
    'scale': 0.45,  # image scale (+/- gain)
    'shear': 0.5,  # image shear (+/- deg)
    'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
    'flipud': 0.15,  # image flip up-down (probability)
    'fliplr': 0.5,  # image flip left-right (probability)
    'mosaic': 0.8,  # image mosaic (probability)
    'mixup': 0.35,  # image mixup (probability)
    'copy_paste': 0,  # image copy paste (probability)
    'paste_in': 0,  # image copy paste (probability), use 0 for faster training
    'loss_ota': 1, # use ComputeLossOTA, use 0 for faster training
}

directory = 'lib/yolov7/data/'
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory {directory} created.")
else:
    print(f"Directory {directory} already exists.")

filepath = os.path.join(directory, 'hyp.yaml')
with open(filepath, 'w') as f:
    yaml.dump(hyp, f)
    print(f"'hyp.yaml' saved at {filepath}")

#print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")



'''''
# colab T4 GPU
python lib/yolov7/train_aux.py \
  --batch-size 10 \
  --img-size 1024 \
  --epochs 55 \
  --name yolov7_mad_12 \
  --adam \
  --data lib/MilitaryDataset/data/mad.yaml \
  --cfg lib/yolov7/cfg/training/yolov7-w6.yaml \
  --weights "" \
  --v5-metric \
  --hyp lib/yolov7/hyp.yaml \
  --project lib/YOLOv7_CheckpointsV1 
  #--resume /content/drive/MyDrive/YOLOv7_Checkpoints/yolov7_mad_122/weights/last.pt
'''''

#!mkdir -p /content/drive/MyDrive/YOLOv7_Checkpoints/detections/guesses
'''''
git tag v0.1

python /content/yolov7/detect.py \
  --weights /content/drive/MyDrive/YOLOv7_Checkpoints/yolov7_mad_122/weights/best.pt \
  --img-size 1024 \
  --conf-thres 0.01 \
  --iou-thres 0.01 \
  --source /content/airliner-airbus-a340.png \
  --project /content/drive/MyDrive/YOLOv7_Checkpoints/ \
  --name detections/guesses \
  --exist-ok
'''''
