# models/detector.py
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import logging
from typing import List, Dict, Any
import os

logger = logging.getLogger(__name__)

class CarDefectDetector:
    """
    Детектор дефектов автомобилей на основе YOLO
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_loaded = False
        self.class_names = {
            0: "rust",
            1: "scratch", 
            2: "dent",
            3: "crack"
        }
        
        # Загружаем модель
        self.load_model(model_path)
    
    def load_model(self, model_path: str = None):
        """
        Загрузка модели YOLO
        """
        try:
            if model_path and os.path.exists(model_path):
                # Загружаем кастомную модель
                self.model = YOLO(model_path)
                logger.info(f"Загружена кастомная модель: {model_path}")
            else:
                # Используем предобученную YOLOv8 модель
                self.model = YOLO('yolov8n.pt')
                logger.info("Загружена предобученная модель YOLOv8n")
            
            self.model_loaded = True
            logger.info("Модель успешно загружена")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {str(e)}")
            self.model_loaded = False
    
    def is_model_loaded(self) -> bool:
        """Проверка загрузки модели"""
        return self.model_loaded
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Обнаружение дефектов на изображении
        
        Args:
            image: Входное изображение (BGR)
            
        Returns:
            Список обнаруженных дефектов
        """
        if not self.model_loaded:
            logger.warning("Модель не загружена")
            return []
        
        try:
            # Выполняем детекцию
            results = self.model(image, conf=0.3, iou=0.5)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Координаты bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Получаем название класса
                        class_name = self.class_names.get(class_id, f"class_{class_id}")
                        
                        detection = {
                            "bbox": {
                                "x1": float(x1),
                                "y1": float(y1), 
                                "x2": float(x2),
                                "y2": float(y2)
                            },
                            "confidence": float(confidence),
                            "class": class_name,
                            "class_id": class_id,
                            "area": float((x2 - x1) * (y2 - y1))
                        }
                        
                        detections.append(detection)
            
            logger.info(f"Обнаружено {len(detections)} дефектов")
            return detections
            
        except Exception as e:
            logger.error(f"Ошибка детекции: {str(e)}")
            return []
    
    def detect_custom(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Кастомная детекция дефектов (для демонстрации)
        В реальном проекте здесь будет обученная модель
        """
        # Простая имитация детекции для демонстрации
        height, width = image.shape[:2]
        
        # Имитируем обнаружение ржавчины и царапин
        detections = []
        
        # Ищем области с низкой яркостью (потенциальная ржавчина)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark_areas = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1]
        
        contours, _ = cv2.findContours(dark_areas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Минимальная площадь
                x, y, w, h = cv2.boundingRect(contour)
                confidence = min(0.9, area / 10000)  # Имитация confidence
                
                detection = {
                    "bbox": {
                        "x1": float(x),
                        "y1": float(y),
                        "x2": float(x + w),
                        "y2": float(y + h)
                    },
                    "confidence": float(confidence),
                    "class": "rust",
                    "class_id": 0,
                    "area": float(area)
                }
                
                detections.append(detection)
        
        # Ищем линейные структуры (потенциальные царапины)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if length > 50:  # Минимальная длина царапины
                    confidence = min(0.8, length / 200)
                    
                    detection = {
                        "bbox": {
                            "x1": float(min(x1, x2)),
                            "y1": float(min(y1, y2)),
                            "x2": float(max(x1, x2)),
                            "y2": float(max(y1, y2))
                        },
                        "confidence": float(confidence),
                        "class": "scratch",
                        "class_id": 1,
                        "area": float(length * 5)  # Примерная площадь
                    }
                    
                    detections.append(detection)
        
        logger.info(f"Кастомная детекция: найдено {len(detections)} дефектов")
        return detections
