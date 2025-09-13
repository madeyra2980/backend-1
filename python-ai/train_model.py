# train_model.py
"""
Скрипт для обучения модели детекции дефектов автомобилей
"""
import os
import yaml
import torch
from ultralytics import YOLO
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dataset_config():
    """
    Создание конфигурации датасета для YOLO
    """
    config = {
        'path': './data',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 4,  # Количество классов
        'names': {
            0: 'rust',
            1: 'scratch', 
            2: 'dent',
            3: 'crack'
        }
    }
    
    with open('data/dataset.yaml', 'w') as f:
        yaml.dump(config, f)
    
    logger.info("Конфигурация датасета создана")
    return config

def prepare_data_structure():
    """
    Подготовка структуры папок для данных
    """
    folders = [
        'data/images/train',
        'data/images/val', 
        'data/images/test',
        'data/labels/train',
        'data/labels/val',
        'data/labels/test'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    logger.info("Структура папок создана")

def train_yolo_model():
    """
    Обучение YOLO модели
    """
    # Создаем структуру данных
    prepare_data_structure()
    create_dataset_config()
    
    # Загружаем предобученную модель
    model = YOLO('yolov8n.pt')
    
    # Параметры обучения
    training_args = {
        'data': 'data/dataset.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'project': 'runs/detect',
        'name': 'car_defects',
        'save': True,
        'save_period': 10,
        'patience': 20,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 2.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0
    }
    
    logger.info("Начинаем обучение модели...")
    logger.info(f"Устройство: {training_args['device']}")
    
    # Запускаем обучение
    results = model.train(**training_args)
    
    logger.info("Обучение завершено!")
    logger.info(f"Лучшая модель сохранена в: {results.save_dir}")
    
    return results

def validate_model(model_path: str):
    """
    Валидация обученной модели
    """
    model = YOLO(model_path)
    
    # Валидация на тестовом наборе
    results = model.val(data='data/dataset.yaml', split='test')
    
    logger.info("Результаты валидации:")
    logger.info(f"mAP50: {results.box.map50}")
    logger.info(f"mAP50-95: {results.box.map}")
    
    return results

def create_sample_data():
    """
    Создание примеров данных для демонстрации
    """
    logger.info("Создание примеров данных...")
    
    # Создаем README с инструкциями
    readme_content = """
# Данные для обучения модели детекции дефектов

## Структура папок:
```
data/
├── images/
│   ├── train/     # Обучающие изображения
│   ├── val/       # Валидационные изображения  
│   └── test/      # Тестовые изображения
└── labels/
    ├── train/     # Аннотации для обучения (YOLO формат)
    ├── val/       # Аннотации для валидации
    └── test/      # Аннотации для тестирования
```

## Формат аннотаций (YOLO):
Каждый файл .txt содержит аннотации для соответствующего изображения:
```
class_id center_x center_y width height
```

Где:
- class_id: 0=rust, 1=scratch, 2=dent, 3=crack
- center_x, center_y: координаты центра (нормализованные 0-1)
- width, height: ширина и высота (нормализованные 0-1)

## Инструменты для разметки:
- LabelImg: https://github.com/tzutalin/labelImg
- Roboflow: https://roboflow.com/
- CVAT: https://github.com/opencv/cvat

## Примеры команд:
```bash
# Обучение модели
python train_model.py

# Валидация
python validate_model.py --model runs/detect/car_defects/weights/best.pt

# Тестирование на изображении
python test_detection.py --image path/to/image.jpg --model runs/detect/car_defects/weights/best.pt
```
"""
    
    with open('data/README.md', 'w') as f:
        f.write(readme_content)
    
    logger.info("Примеры данных созданы")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Обучение модели детекции дефектов')
    parser.add_argument('--mode', choices=['train', 'validate', 'prepare'], 
                       default='prepare', help='Режим работы')
    parser.add_argument('--model', type=str, help='Путь к модели для валидации')
    
    args = parser.parse_args()
    
    if args.mode == 'prepare':
        create_sample_data()
    elif args.mode == 'train':
        train_yolo_model()
    elif args.mode == 'validate':
        if not args.model:
            logger.error("Укажите путь к модели с помощью --model")
        else:
            validate_model(args.model)
