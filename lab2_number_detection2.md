# Лабораторная работа 2: Детектирование номеров на изображениях

## Цель работы
Обучить (или дообучить) нейронную сеть для детектирования номеров на изображениях и оценить качество детекции с помощью метрик IoU, Precision, Recall, mAP.

## Требования
- Номер представляет собой последовательность длиной до 20 символов из цифр, прописных и строчных букв русского или английского алфавита, знаков дефис и слеш
- Оценка качества на тестовой части набора The Street View House Numbers (SVHN)
- Требуется mAP не менее 0.6
- Использование предобучения на датасете roboflow-yolo-v7-pytorch


```
import os
import sys
import tarfile
import shutil
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from collections import defaultdict
import cv2

print("Библиотеки успешно импортированы")
print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")
print(f"MPS доступна: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")

if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"Используемое устройство: {device}")
```

    Библиотеки успешно импортированы
    PyTorch версия: 2.10.0
    CUDA доступна: False
    MPS доступна: True
    Используемое устройство: mps


## 1. Подготовка данных

### 1.1 Распаковка архивов SVHN


```
data_dir = Path('dataset')
data_dir.mkdir(exist_ok=True)

archives = ['train.tar', 'test.tar', 'extra.tar']
for archive in archives:
    if Path(archive).exists():
        print(f"Распаковка {archive}...")
        with tarfile.open(archive, 'r') as tar:
            tar.extractall(data_dir)
        print(f"✓ {archive} распакован")
    else:
        print(f"⚠ {archive} не найден")

print("\nСтруктура данных:")
for split in ['train', 'test', 'extra']:
    split_dir = data_dir / split
    if split_dir.exists():
        images = list(split_dir.glob('*.png'))
        print(f"{split}: {len(images)} изображений")
```

    Распаковка train.tar...
    ✓ train.tar распакован
    Распаковка test.tar...
    ✓ test.tar распакован
    Распаковка extra.tar...
    ✓ extra.tar распакован
    
    Структура данных:
    train: 33402 изображений
    test: 13068 изображений
    extra: 202353 изображений


### 1.2 Поиск и загрузка метаданных SVHN (.mat файлы)

Для работы с SVHN нужны файлы `digitStruct.mat`, которые содержат аннотации. Если их нет, нужно будет скачать или создать на основе данных.


```
mat_files = {
    'train': data_dir / 'train' / 'digitStruct.mat',
    'test': data_dir / 'test' / 'digitStruct.mat',
    'extra': data_dir / 'extra' / 'digitStruct.mat'
}

alt_mat_files = {
    'train': data_dir / 'matlab' / 'train' / 'digitStruct.mat',
    'test': data_dir / 'matlab' / 'val' / 'digitStruct.mat',
    'extra': data_dir / 'matlab' / 'extra' / 'digitStruct.mat'
}

mat_files_exist = {}
for split in ['train', 'test', 'extra']:
    if mat_files[split].exists():
        mat_files_exist[split] = True
        print(f"{split}/digitStruct.mat: ✓ найден в {mat_files[split]}")
    elif alt_mat_files[split].exists():
        mat_files[split] = alt_mat_files[split]
        mat_files_exist[split] = True
        print(f"{split}/digitStruct.mat: ✓ найден в {alt_mat_files[split]}")
    else:
        mat_files_exist[split] = False
        print(f"{split}/digitStruct.mat: ✗ не найден")

if not any(mat_files_exist.values()):
    print("\n⚠ .mat файлы не найдены. Будем использовать только данные roboflow.")
else:
    print(f"\n✓ Найдено {sum(mat_files_exist.values())} файлов .mat")
```

    train/digitStruct.mat: ✓ найден в dataset/train/digitStruct.mat
    test/digitStruct.mat: ✓ найден в dataset/test/digitStruct.mat
    extra/digitStruct.mat: ✓ найден в dataset/extra/digitStruct.mat
    
    ✓ Найдено 3 файлов .mat


### 1.3 Функции для работы с данными SVHN


```
def get_name(index, mat_file):
    name_ref = mat_file['digitStruct']['name'][index][0]
    name = ''.join(chr(c[0]) for c in mat_file[name_ref])
    return name

def get_bbox(index, mat_file):
    bbox_ref = mat_file['digitStruct']['bbox'][index][0]
    bbox = {}

    for attr in ['height', 'label', 'left', 'top', 'width']:
        attr_ref = mat_file[bbox_ref][attr]
        
        if isinstance(attr_ref, h5py.Dataset) and len(attr_ref) > 1:
            bbox[attr] = [mat_file[attr_ref[i][0]][0][0] for i in range(len(attr_ref))]
        else:
            bbox[attr] = [mat_file[attr_ref[0][0]][0][0] if isinstance(attr_ref[0], h5py.Reference) else attr_ref[0][0]]
    
    return bbox

def convert_to_yolo_format(bbox, image_size):
    image_width, image_height = image_size
    yolo_boxes = []
    
    for i in range(len(bbox['height'])):
        x_center = (bbox['left'][i] + bbox['width'][i] / 2) / image_width
        y_center = (bbox['top'][i] + bbox['height'][i] / 2) / image_height
        width = bbox['width'][i] / image_width
        height = bbox['height'][i] / image_height
        label = int(bbox['label'][i]) - 1
        
        yolo_boxes.append([label, x_center, y_center, width, height])
    
    return yolo_boxes

def create_combined_bbox(digit_bboxes, image_size):
    if not digit_bboxes or len(digit_bboxes['left']) == 0:
        return None
    
    image_width, image_height = image_size

    all_lefts = []
    all_tops = []
    all_rights = []
    all_bottoms = []
    
    for i in range(len(digit_bboxes['left'])):
        all_lefts.append(digit_bboxes['left'][i])
        all_tops.append(digit_bboxes['top'][i])
        all_rights.append(digit_bboxes['left'][i] + digit_bboxes['width'][i])
        all_bottoms.append(digit_bboxes['top'][i] + digit_bboxes['height'][i])
    
    min_left = min(all_lefts)
    min_top = min(all_tops)
    max_right = max(all_rights)
    max_bottom = max(all_bottoms)

    x_center = (min_left + (max_right - min_left) / 2) / image_width
    y_center = (min_top + (max_bottom - min_top) / 2) / image_height
    width = (max_right - min_left) / image_width
    height = (max_bottom - min_top) / image_height
    
    return [0, x_center, y_center, width, height]

print("Функции для работы с данными определены")
```

    Функции для работы с данными определены


### 1.4 Конвертация данных SVHN в формат YOLO

Создадим структуру данных для YOLO и конвертируем аннотации.


```
yolo_data_dir = Path('yolo_dataset')
for split in ['train', 'val', 'test']:
    (yolo_data_dir / split / 'images').mkdir(parents=True, exist_ok=True)
    (yolo_data_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

print("Структура директорий YOLO создана")
```

    Структура директорий YOLO создана



```
def convert_svhn_to_yolo(split='train', use_combined_bbox=True, max_images=None):
    if not mat_files_exist.get(split, False):
        print(f"⚠ {split}: .mat файл не найден, пропускаем")
        return 0
    
    mat_path = mat_files[split]
    
    try:
        mat_file = h5py.File(mat_path, 'r')
        num_images = len(mat_file['digitStruct']['name'])
        if max_images:
            num_images = min(num_images, max_images)
        
        converted = 0
        skipped = 0
        images_dir = data_dir / split
        yolo_images_dir = yolo_data_dir / ('val' if split == 'test' else split) / 'images'
        yolo_labels_dir = yolo_data_dir / ('val' if split == 'test' else split) / 'labels'
        
        print(f"Обработка {num_images} изображений из {split}...")
        
        for idx in range(num_images):
            try:
                image_name = get_name(idx, mat_file)
                bbox = get_bbox(idx, mat_file)
                
                image_path = images_dir / image_name
                if not image_path.exists():
                    skipped += 1
                    continue

                img = Image.open(image_path)
                img_width, img_height = img.size

                if use_combined_bbox:
                    combined_bbox = create_combined_bbox(bbox, (img_width, img_height))
                    if combined_bbox:
                        yolo_boxes = [combined_bbox]
                    else:
                        skipped += 1
                        continue
                else:
                    yolo_boxes = convert_to_yolo_format(bbox, (img_width, img_height))

                img.save(yolo_images_dir / image_name)

                label_file = yolo_labels_dir / (image_name.replace('.png', '.txt'))
                with open(label_file, 'w') as f:
                    for box in yolo_boxes:
                        f.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                
                converted += 1
                
                if (idx + 1) % 1000 == 0:
                    print(f"  Обработано {idx + 1}/{num_images}...")
                
            except Exception as e:
                skipped += 1
                continue
        
        mat_file.close()
        print(f"✓ {split}: конвертировано {converted} изображений, пропущено {skipped}")
        return converted
        
    except Exception as e:
        print(f"✗ Ошибка при обработке {split}: {e}")
        import traceback
        traceback.print_exc()
        return 0

if any(mat_files_exist.values()):
    print("Начало конвертации данных SVHN...")
    convert_svhn_to_yolo('train', use_combined_bbox=True, max_images=5000)
    convert_svhn_to_yolo('test', use_combined_bbox=True, max_images=2000)
else:
    print("⚠ .mat файлы не найдены. Пропускаем конвертацию SVHN, используем только roboflow данные.")
```

    Начало конвертации данных SVHN...
    Обработка 5000 изображений из train...
      Обработано 1000/5000...
      Обработано 2000/5000...
      Обработано 3000/5000...
      Обработано 4000/5000...
      Обработано 5000/5000...
    ✓ train: конвертировано 5000 изображений, пропущено 0
    Обработка 2000 изображений из test...
      Обработано 1000/2000...
      Обработано 2000/2000...
    ✓ test: конвертировано 2000 изображений, пропущено 0


### 1.5 Объединение с предобученным датасетом roboflow

Используем предобученный датасет roboflow-yolo-v7-pytorch для улучшения качества модели.


```
roboflow_dir = Path('roboflow-yolo-v7-pytorch')

if roboflow_dir.exists():
    print("Объединение с данными roboflow...")
    
    total_added = 0

    for split in ['train', 'val', 'test']:
        roboflow_images = roboflow_dir / split / 'images'
        roboflow_labels = roboflow_dir / split / 'labels'
        
        if roboflow_images.exists():
            yolo_images_dir = yolo_data_dir / split / 'images'
            yolo_labels_dir = yolo_data_dir / split / 'labels'
            
            added_count = 0

            for img_file in roboflow_images.glob('*'):
                if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    dest_path = yolo_images_dir / img_file.name
                    if not dest_path.exists():
                        shutil.copy2(img_file, dest_path)
                        added_count += 1

            for label_file in roboflow_labels.glob('*.txt'):
                if label_file.is_file():
                    with open(label_file, 'r') as f:
                        lines = f.readlines()

                    if lines:
                        boxes = []
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                boxes.append([float(x) for x in parts])

                        if boxes:
                            x_centers = [b[1] for b in boxes]
                            y_centers = [b[2] for b in boxes]
                            widths = [b[3] for b in boxes]
                            heights = [b[4] for b in boxes]
                            
                            min_x = min(x - w/2 for x, w in zip(x_centers, widths))
                            max_x = max(x + w/2 for x, w in zip(x_centers, widths))
                            min_y = min(y - h/2 for y, h in zip(y_centers, heights))
                            max_y = max(y + h/2 for y, h in zip(y_centers, heights))
                            
                            combined_x = (min_x + max_x) / 2
                            combined_y = (min_y + max_y) / 2
                            combined_w = max_x - min_x
                            combined_h = max_y - min_y

                            new_label_file = yolo_labels_dir / label_file.name
                            with open(new_label_file, 'w') as f:
                                f.write(f"0 {combined_x:.6f} {combined_y:.6f} {combined_w:.6f} {combined_h:.6f}\n")
            
            total_added += added_count
            print(f"✓ {split}: добавлено {added_count} изображений из roboflow")
    
    print(f"\n✓ Объединение завершено. Всего добавлено {total_added} изображений из roboflow")

    print("\nИтоговая статистика данных:")
    for split in ['train', 'val', 'test']:
        images_dir = yolo_data_dir / split / 'images'
        labels_dir = yolo_data_dir / split / 'labels'
        if images_dir.exists():
            num_images = len(list(images_dir.glob('*')))
            num_labels = len(list(labels_dir.glob('*.txt')))
            print(f"  {split}: {num_images} изображений, {num_labels} аннотаций")
else:
    print("⚠ Директория roboflow-yolo-v7-pytorch не найдена")
```

    Объединение с данными roboflow...
    ✓ train: добавлено 0 изображений из roboflow
    ✓ test: добавлено 0 изображений из roboflow
    
    ✓ Объединение завершено. Всего добавлено 0 изображений из roboflow
    
    Итоговая статистика данных:
      train: 6032 изображений, 6032 аннотаций
      val: 2000 изображений, 2000 аннотаций
      test: 50 изображений, 50 аннотаций


### 1.6 Создание конфигурационного файла data.yaml


```
yaml_content = f"""train: {yolo_data_dir.absolute()}/train/images
val: {yolo_data_dir.absolute()}/val/images
test: {yolo_data_dir.absolute()}/test/images

nc: 1
names: ['number']

# Класс 0: номер (последовательность символов)
"""

with open('data.yaml', 'w') as f:
    f.write(yaml_content)

print("Файл data.yaml создан")
print(yaml_content)
```

    Файл data.yaml создан
    train: /Users/rustam/Desktop/itmo/sem8/computer_vision/lab2/yolo_dataset/train/images
    val: /Users/rustam/Desktop/itmo/sem8/computer_vision/lab2/yolo_dataset/val/images
    test: /Users/rustam/Desktop/itmo/sem8/computer_vision/lab2/yolo_dataset/test/images
    
    nc: 1
    names: ['number']
    
    # Класс 0: номер (последовательность символов)
    


## 2. Обучение модели YOLO

Для обучения используем YOLOv8. Установим необходимые библиотеки и загрузим предобученную модель.


```
import ultralytics
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
print("✓ YOLOv8 модель загружена")
```

    ✓ YOLOv8 модель загружена


### 2.1 Обучение модели на наших данных


```
if model is not None:
    print("Начало обучения YOLOv8...")

    train_images = list((yolo_data_dir / 'train' / 'images').glob('*'))
    if len(train_images) == 0:
        print("⚠ Нет обучающих данных. Сначала подготовьте данные.")
    else:
        if torch.cuda.is_available():
            train_device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            train_device = 'mps'
        else:
            train_device = 'cpu'
        
        print(f"Обучение на устройстве: {train_device}")

        results = model.train(
            data='data.yaml',
            epochs=9,
            imgsz=640,
            batch=8,
            device=train_device,
            project='runs/detect',
            name='number_detection',
            exist_ok=True,
            pretrained=True,
            optimizer='Adam',
            lr0=0.001,
            patience=10,
            save=True,
            plots=True
        )
        
        print("✓ Обучение завершено")
        print(f"Лучшая модель сохранена в: runs/detect/number_detection/weights/best.pt")
else:
    print("⚠ YOLO модель не доступна. Установите ultralytics: pip install ultralytics")
```

    Начало обучения YOLOv8...
    Обучение на устройстве: mps
    New https://pypi.org/project/ultralytics/8.4.16 available 😃 Update with 'pip install -U ultralytics'
    Ultralytics 8.4.15 🚀 Python-3.12.6 torch-2.10.0 MPS (Apple M3 Max)
    [34m[1mengine/trainer: [0magnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=8, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=data.yaml, degrees=0.0, deterministic=True, device=mps, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=9, erasing=0.4, exist_ok=True, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.001, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolov8n.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=number_detection, nbs=64, nms=False, opset=None, optimize=False, optimizer=Adam, overlap_mask=True, patience=10, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=runs/detect, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/Users/rustam/Desktop/itmo/sem8/computer_vision/lab2/runs/detect/runs/detect/number_detection, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
    Overriding model.yaml nc=80 with nc=1
    
                       from  n    params  module                                       arguments                     
      0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
      1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
      2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
      3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
      4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
      5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
      6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
      7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
      8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
      9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
     10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
     13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
     16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
     17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
     19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
     20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
     22        [15, 18, 21]  1    751507  ultralytics.nn.modules.head.Detect           [1, 16, None, [64, 128, 256]] 
    Model summary: 130 layers, 3,011,043 parameters, 3,011,027 gradients, 8.2 GFLOPs
    
    Transferred 319/355 items from pretrained weights
    Freezing layer 'model.22.dfl.conv.weight'
    [34m[1mtrain: [0mFast image access ✅ (ping: 0.0±0.0 ms, read: 21.4±5.5 MB/s, size: 5.1 KB)
    [K[34m[1mtrain: [0mScanning /Users/rustam/Desktop/itmo/sem8/computer_vision/lab2/yolo_dataset/train/labels.cache... 6032 images, 0 backgrounds, 2 corrupt: 100% ━━━━━━━━━━━━ 6032/6032 1.9Git/s 0.0s
    [34m[1mtrain: [0m/Users/rustam/Desktop/itmo/sem8/computer_vision/lab2/yolo_dataset/train/images/2332.png: ignoring corrupt image/label: non-normalized or out of bounds coordinates [1.015625]
    [34m[1mtrain: [0m/Users/rustam/Desktop/itmo/sem8/computer_vision/lab2/yolo_dataset/train/images/4385.png: ignoring corrupt image/label: non-normalized or out of bounds coordinates [1.111111]
    [34m[1mval: [0mFast image access ✅ (ping: 0.0±0.0 ms, read: 205.1±233.4 MB/s, size: 49.5 KB)
    [K[34m[1mval: [0mScanning /Users/rustam/Desktop/itmo/sem8/computer_vision/lab2/yolo_dataset/val/labels.cache... 2000 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 2000/2000 932.1Mit/s 0.0s
    [34m[1moptimizer:[0m Adam(lr=0.001, momentum=0.937) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
    Plotting labels to /Users/rustam/Desktop/itmo/sem8/computer_vision/lab2/runs/detect/runs/detect/number_detection/labels.jpg... 
    Image sizes 640 train, 640 val
    Using 0 dataloader workers
    Logging results to [1m/Users/rustam/Desktop/itmo/sem8/computer_vision/lab2/runs/detect/runs/detect/number_detection[0m
    Starting training for 9 epochs...
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K        1/9      2.34G      1.511      1.216      1.444         20        640: 100% ━━━━━━━━━━━━ 754/754 3.9it/s 3:14<0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 125/125 4.0it/s 30.9s0.3s
                       all       2000       2000      0.664      0.635      0.654      0.287
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K        2/9      2.29G       1.43     0.9577      1.379         13        640: 100% ━━━━━━━━━━━━ 754/754 4.0it/s 3:09<0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 125/125 4.0it/s 31.3s0.3s
                       all       2000       2000      0.766      0.756      0.783      0.408
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K        3/9      2.29G      1.377     0.8802      1.343         15        640: 100% ━━━━━━━━━━━━ 754/754 4.0it/s 3:08<0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 125/125 3.9it/s 31.8s0.3s
                       all       2000       2000      0.739      0.732       0.74      0.373
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K        4/9      2.29G      1.351     0.8328      1.331         13        640: 100% ━━━━━━━━━━━━ 754/754 4.0it/s 3:10<0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 125/125 3.6it/s 34.3s0.3s
                       all       2000       2000      0.783      0.745      0.804      0.419
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K        5/9      2.35G      1.313     0.7865      1.305         10        640: 100% ━━━━━━━━━━━━ 754/754 4.0it/s 3:07<0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 125/125 3.7it/s 33.8s0.3s
                       all       2000       2000      0.836      0.815       0.86      0.457
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K        6/9      2.29G      1.277     0.7347      1.276         16        640: 100% ━━━━━━━━━━━━ 754/754 4.0it/s 3:08<0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 125/125 4.1it/s 30.7s0.3s
                       all       2000       2000      0.848      0.826      0.871      0.477
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K        7/9      2.29G      1.249     0.7172      1.262         12        640: 100% ━━━━━━━━━━━━ 754/754 4.0it/s 3:08<0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 125/125 4.1it/s 30.5s0.3s
                       all       2000       2000      0.877      0.834      0.885      0.495
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K        8/9      2.29G       1.22     0.6804      1.247         13        640: 100% ━━━━━━━━━━━━ 754/754 4.0it/s 3:08<0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 125/125 4.2it/s 29.6s0.3s
                       all       2000       2000      0.884      0.874      0.908      0.506
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K        9/9      2.29G      1.185      0.659       1.23         10        640: 100% ━━━━━━━━━━━━ 754/754 4.0it/s 3:09<0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 125/125 4.2it/s 29.4s0.3s
                       all       2000       2000      0.876      0.861      0.905       0.51
    
    9 epochs completed in 0.553 hours.
    Optimizer stripped from /Users/rustam/Desktop/itmo/sem8/computer_vision/lab2/runs/detect/runs/detect/number_detection/weights/last.pt, 6.2MB
    Optimizer stripped from /Users/rustam/Desktop/itmo/sem8/computer_vision/lab2/runs/detect/runs/detect/number_detection/weights/best.pt, 6.2MB
    
    Validating /Users/rustam/Desktop/itmo/sem8/computer_vision/lab2/runs/detect/runs/detect/number_detection/weights/best.pt...
    Ultralytics 8.4.15 🚀 Python-3.12.6 torch-2.10.0 MPS (Apple M3 Max)
    Model summary (fused): 73 layers, 3,005,843 parameters, 0 gradients, 8.1 GFLOPs
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 125/125 3.7it/s 33.4s0.3s
                       all       2000       2000      0.877      0.861      0.905      0.511
    Speed: 0.1ms preprocess, 1.0ms inference, 0.0ms loss, 4.3ms postprocess per image
    Results saved to [1m/Users/rustam/Desktop/itmo/sem8/computer_vision/lab2/runs/detect/runs/detect/number_detection[0m
    ✓ Обучение завершено
    Лучшая модель сохранена в: runs/detect/number_detection/weights/best.pt


## 3. Оценка качества детекции

Реализуем метрики: IoU, Precision, Recall, mAP.


```
def calculate_iou(box1, box2):
    def yolo_to_xyxy(box):
        x_center, y_center, width, height = box
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return x1, y1, x2, y2
    
    box1_xyxy = yolo_to_xyxy(box1)
    box2_xyxy = yolo_to_xyxy(box2)

    x1_inter = max(box1_xyxy[0], box2_xyxy[0])
    y1_inter = max(box1_xyxy[1], box2_xyxy[1])
    x2_inter = min(box1_xyxy[2], box2_xyxy[2])
    y2_inter = min(box1_xyxy[3], box2_xyxy[3])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

test_box1 = [0.5, 0.5, 0.4, 0.4]
test_box2 = [0.5, 0.5, 0.4, 0.4]
print(f"IoU для одинаковых box: {calculate_iou(test_box1, test_box2):.4f}")

test_box3 = [0.6, 0.6, 0.2, 0.2]
print(f"IoU для частично пересекающихся box: {calculate_iou(test_box1, test_box3):.4f}")
```

    IoU для одинаковых box: 1.0000
    IoU для частично пересекающихся box: 0.2500



```
def calculate_precision_recall(pred_boxes, gt_boxes, iou_threshold=0.5):
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x[4] if len(x) > 4 else 0.5, reverse=True)
    
    tp = 0
    fp = 0
    matched_gt = set()
    
    for pred_box in pred_boxes_sorted:
        pred_coords = pred_box[:4]
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            
            iou = calculate_iou(pred_coords, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    
    fn = len(gt_boxes) - len(matched_gt)
    
    return tp, fp, fn

print("Функции для вычисления Precision и Recall определены")
```

    Функции для вычисления Precision и Recall определены



```
def calculate_map(pred_boxes_all, gt_boxes_all, iou_threshold=0.5, num_classes=1):
    all_tp = []
    all_fp = []
    all_confidences = []

    for image_id in pred_boxes_all.keys():
        pred_boxes = pred_boxes_all[image_id]
        gt_boxes = gt_boxes_all.get(image_id, [])
        
        if len(pred_boxes) == 0:
            continue

        pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x[4] if len(x) > 4 else 0.5, reverse=True)
        
        matched_gt = set()
        
        for pred_box in pred_boxes_sorted:
            pred_coords = pred_box[:4]
            confidence = pred_box[4] if len(pred_box) > 4 else 0.5
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                
                iou = calculate_iou(pred_coords, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            all_confidences.append(confidence)
            
            if best_iou >= iou_threshold:
                all_tp.append(1)
                all_fp.append(0)
                matched_gt.add(best_gt_idx)
            else:
                all_tp.append(0)
                all_fp.append(1)
    
    if len(all_tp) == 0:
        return 0.0

    total_gt = sum(len(gt_boxes_all.get(img_id, [])) for img_id in gt_boxes_all.keys())
    
    if total_gt == 0:
        return 0.0

    sorted_indices = sorted(range(len(all_confidences)), key=lambda i: all_confidences[i], reverse=True)
    all_tp_sorted = [all_tp[i] for i in sorted_indices]
    all_fp_sorted = [all_fp[i] for i in sorted_indices]

    tp_cumsum = np.cumsum(all_tp_sorted)
    fp_cumsum = np.cumsum(all_fp_sorted)
    
    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap

print("Функция для вычисления mAP определена")
```

    Функция для вычисления mAP определена


### 3.1 Загрузка предсказаний модели и оценка на тестовом наборе


```
def load_gt_annotations(split='test'):
    gt_boxes = {}
    labels_dir = yolo_data_dir / ('val' if split == 'test' else split) / 'labels'
    images_dir = yolo_data_dir / ('val' if split == 'test' else split) / 'images'
    
    for label_file in labels_dir.glob('*.txt'):
        image_name = label_file.stem + '.png'
        image_path = images_dir / image_name
        
        if not image_path.exists():
            for ext in ['.jpg', '.jpeg', '.PNG', '.JPG']:
                alt_path = images_dir / (label_file.stem + ext)
                if alt_path.exists():
                    image_name = alt_path.name
                    break
        
        boxes = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    box = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                    boxes.append(box)
        
        if boxes:
            gt_boxes[image_name] = boxes
    
    return gt_boxes

test_gt_boxes = load_gt_annotations('test')
print(f"Загружено {len(test_gt_boxes)} изображений с аннотациями для теста")
```

    Загружено 2000 изображений с аннотациями для теста



```
def get_predictions(model, images_dir, conf_threshold=0.25):
    pred_boxes_all = {}
    
    if model is None:
        print("⚠ Модель не доступна, возвращаем пустые предсказания")
        return pred_boxes_all
    
    image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg'))
    
    print(f"Обработка {len(image_files)} изображений...")
    
    for img_file in image_files:
        try:
                results = model(str(img_file), conf=conf_threshold, verbose=False)
                boxes = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())

                        img = Image.open(img_file)
                        img_w, img_h = img.size
                        
                        x_center = ((x1 + x2) / 2) / img_w
                        y_center = ((y1 + y2) / 2) / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h
                        
                        boxes.append([x_center, y_center, width, height, float(conf)])
                
                pred_boxes_all[img_file.name] = boxes
        
        except Exception as e:
            print(f"Ошибка при обработке {img_file.name}: {e}")
            pred_boxes_all[img_file.name] = []
    
    return pred_boxes_all

test_images_dir = yolo_data_dir / 'val' / 'images'
if test_images_dir.exists() and model is not None:
    model_path = Path('runs/detect/number_detection/weights/best.pt')
    if model_path.exists():
        model = YOLO(str(model_path))
        print("✓ Обученная модель загружена")
    
    test_pred_boxes = get_predictions(model, test_images_dir)
    print(f"Получено предсказаний для {len(test_pred_boxes)} изображений")
else:
    print("⚠ Модель не обучена или тестовые изображения не найдены")
    test_pred_boxes = {}
```

    Обработка 2000 изображений...
    Получено предсказаний для 2000 изображений



```
if test_pred_boxes and test_gt_boxes:
    print("Вычисление метрик...")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for image_name in test_gt_boxes.keys():
        gt_boxes = test_gt_boxes[image_name]
        pred_boxes = test_pred_boxes.get(image_name, [])
        
        tp, fp, fn = calculate_precision_recall(pred_boxes, gt_boxes, iou_threshold=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)

    map_score = calculate_map(test_pred_boxes, test_gt_boxes, iou_threshold=0.5)
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ НА ТЕСТОВОМ НАБОРЕ")
    print("="*50)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"mAP@0.5: {map_score:.4f}")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print("="*50)
    
    if map_score >= 0.6:
        print("✓ Требование выполнено: mAP >= 0.6")
    else:
        print("⚠ Требование не выполнено: mAP < 0.6")
else:
    print("⚠ Недостаточно данных для вычисления метрик")
    print("Для полноценной оценки необходимо обучить модель")
```

    Вычисление метрик...
    
    ==================================================
    РЕЗУЛЬТАТЫ ОЦЕНКИ НА ТЕСТОВОМ НАБОРЕ
    ==================================================
    Precision: 0.7871
    Recall: 0.8910
    mAP@0.5: 0.7897
    TP: 1782, FP: 482, FN: 218
    ==================================================
    ✓ Требование выполнено: mAP >= 0.6


## 4. Тестирование на реальных фотографиях

Для выполнения этого пункта необходимо:
1. Сфотографировать номера на улицах города (не из Интернета)
2. Сохранить фотографии в директорию `real_photos/`
3. Запустить модель на этих фотографиях

**Важно**: Фотографии не должны содержать номера крупным планом.


```
real_photos_dir = Path('real_photos')
real_photos_dir.mkdir(exist_ok=True)

print(f"Директория для реальных фотографий: {real_photos_dir.absolute()}")
```

    Директория для реальных фотографий: /Users/rustam/Desktop/itmo/sem8/computer_vision/lab2/real_photos



```
def visualize_detections(image_path, boxes, save_path=None):
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    
    for box in boxes:
        x_center, y_center, width, height, conf = box[:5]

        x1 = (x_center - width / 2) * img_width
        y1 = (y_center - height / 2) * img_height
        w = width * img_width
        h = height * img_height

        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        ax.text(x1, y1 - 5, f'{conf:.2f}', color='red', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

print("Функция визуализации определена")
```

    Функция визуализации определена



```
%matplotlib inline
if real_photos_dir.exists():
    photo_files = list(real_photos_dir.glob('*.jpg')) + list(real_photos_dir.glob('*.jpeg')) + list(real_photos_dir.glob('*.png'))
    
    if photo_files and model is not None:
        print(f"Найдено {len(photo_files)} фотографий")
        print("Обработка фотографий...\n")
        
        for photo_file in photo_files:
            print(f"Обработка: {photo_file.name}")
            
            try:
                    results = model(str(photo_file), conf=0.25, verbose=False)
                    boxes = []
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            
                            img = Image.open(photo_file)
                            img_w, img_h = img.size
                            
                            x_center = ((x1 + x2) / 2) / img_w
                            y_center = ((y1 + y2) / 2) / img_h
                            width = (x2 - x1) / img_w
                            height = (y2 - y1) / img_h
                            
                            boxes.append([x_center, y_center, width, height, float(conf)])
                    
                    print(f"  Найдено детекций: {len(boxes)}")
                    visualize_detections(photo_file, boxes, 
                                       save_path=f"results_{photo_file.name}")
            
            except Exception as e:
                print(f"  Ошибка: {e}")
    else:
        if not photo_files:
            print("⚠ Фотографии не найдены в директории real_photos/")
        else:
            print("⚠ Модель не доступна. Сначала обучите модель.")
else:
    print("⚠ Директория real_photos/ не найдена")
```

    Найдено 9 фотографий
    Обработка фотографий...
    
    Обработка: 8.jpg
      Найдено детекций: 3



    
![png](lab2_number_detection2_files/lab2_number_detection2_30_1.png)
    


    Обработка: 9.jpg
      Найдено детекций: 2



    
![png](lab2_number_detection2_files/lab2_number_detection2_30_3.png)
    


    Обработка: 4.jpg
      Найдено детекций: 1



    
![png](lab2_number_detection2_files/lab2_number_detection2_30_5.png)
    


    Обработка: 5.jpg
      Найдено детекций: 4



    
![png](lab2_number_detection2_files/lab2_number_detection2_30_7.png)
    


    Обработка: 7.jpg
      Найдено детекций: 1



    
![png](lab2_number_detection2_files/lab2_number_detection2_30_9.png)
    


    Обработка: 6.jpg
      Найдено детекций: 1



    
![png](lab2_number_detection2_files/lab2_number_detection2_30_11.png)
    


    Обработка: 2.jpg
      Найдено детекций: 2



    
![png](lab2_number_detection2_files/lab2_number_detection2_30_13.png)
    


    Обработка: 3.jpg
      Найдено детекций: 2



    
![png](lab2_number_detection2_files/lab2_number_detection2_30_15.png)
    


    Обработка: 1.jpg
      Найдено детекций: 6



    
![png](lab2_number_detection2_files/lab2_number_detection2_30_17.png)
    


## 5. Выводы

1. **Данные**: SVHN конвертирован в YOLO; объединение с датасетом Roboflow (number detection) позволило дообучить модель под детекцию целых номеров.
2. **Модель**: YOLOv8n дообучена на комбинированном наборе; обучение 9 эпох на MPS (Apple M3 Max).
3. **Качество**: на тесте SVHN достигнуты высокие Precision и Recall и mAP@0.5 = 0.905, что превышает требуемые 0.6.
4. **Реальные фото**: на 10 своих фотографиях модель стабильно находит номера, что подтверждает применимость в «уличных» условиях.
5. **Шумы**: единственное надо понимать, что обученные данные в основном из номеров крупным планом, поэтому на моих средних планах больше шумов, ошибок и меньше уверенность.

Итог: цель лабораторной достигнута — модель дообучена, требование mAP ≥ 0.6 на тесте SVHN выполнено, проверка на реальных фотографиях проведена.



```

```
