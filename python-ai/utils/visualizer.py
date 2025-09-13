# utils/visualizer.py
import cv2
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DetectionVisualizer:
    """
    Класс для визуализации результатов детекции
    """
    
    def __init__(self):
        # Цвета для разных типов дефектов
        self.colors = {
            "rust": (0, 0, 255),      # Красный
            "scratch": (0, 255, 255),  # Желтый
            "dent": (255, 0, 0),      # Синий
            "crack": (0, 255, 0)      # Зеленый
        }
        
        # Названия дефектов на русском
        self.class_names_ru = {
            "rust": "Ржавчина",
            "scratch": "Царапина", 
            "dent": "Вмятина",
            "crack": "Трещина"
        }
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]], 
                       confidence_threshold: float = 0.3) -> np.ndarray:
        """
        Отрисовка обнаруженных дефектов на изображении
        
        Args:
            image: Исходное изображение
            detections: Список обнаруженных дефектов
            confidence_threshold: Минимальный порог уверенности
            
        Returns:
            Изображение с нарисованными дефектами
        """
        result_image = image.copy()
        
        for detection in detections:
            if detection["confidence"] < confidence_threshold:
                continue
                
            bbox = detection["bbox"]
            class_name = detection["class"]
            confidence = detection["confidence"]
            
            # Координаты bounding box
            x1 = int(bbox["x1"])
            y1 = int(bbox["y1"])
            x2 = int(bbox["x2"])
            y2 = int(bbox["y2"])
            
            # Цвет для данного типа дефекта
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Рисуем прямоугольник
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Подготавливаем текст
            label = f"{self.class_names_ru.get(class_name, class_name)}: {confidence:.2f}"
            
            # Размер текста
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Рисуем фон для текста
            cv2.rectangle(result_image, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            
            # Рисуем текст
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Добавляем общую информацию
        self._draw_info_panel(result_image, detections, confidence_threshold)
        
        return result_image
    
    def _draw_info_panel(self, image: np.ndarray, detections: List[Dict[str, Any]], 
                        confidence_threshold: float):
        """
        Отрисовка информационной панели
        """
        height, width = image.shape[:2]
        
        # Подсчитываем статистику
        total_defects = len(detections)
        high_conf_defects = len([d for d in detections if d["confidence"] >= confidence_threshold])
        
        defect_counts = {}
        for detection in detections:
            if detection["confidence"] >= confidence_threshold:
                class_name = detection["class"]
                defect_counts[class_name] = defect_counts.get(class_name, 0) + 1
        
        # Создаем панель информации
        panel_height = 100
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        panel[:] = (50, 50, 50)  # Темно-серый фон
        
        # Текст информации
        info_text = f"Всего дефектов: {total_defects} | Высокая уверенность: {high_conf_defects}"
        cv2.putText(panel, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Детали по типам дефектов
        y_offset = 50
        for class_name, count in defect_counts.items():
            ru_name = self.class_names_ru.get(class_name, class_name)
            color = self.colors.get(class_name, (255, 255, 255))
            text = f"{ru_name}: {count}"
            cv2.putText(panel, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
        
        # Добавляем панель к изображению
        result = np.vstack([image, panel])
        return result
    
    def create_summary_image(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Создание сводного изображения с результатами
        """
        # Создаем сетку 2x2
        height, width = image.shape[:2]
        
        # Оригинальное изображение
        original = cv2.resize(image, (width//2, height//2))
        
        # Изображение с детекциями
        detected = self.draw_detections(image, detections)
        detected = cv2.resize(detected, (width//2, height//2))
        
        # Тепловая карта дефектов
        heatmap = self._create_heatmap(image, detections)
        heatmap = cv2.resize(heatmap, (width//2, height//2))
        
        # Статистическая панель
        stats = self._create_stats_panel(width//2, height//2, detections)
        
        # Объединяем в сетку
        top_row = np.hstack([original, detected])
        bottom_row = np.hstack([heatmap, stats])
        summary = np.vstack([top_row, bottom_row])
        
        return summary
    
    def _create_heatmap(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Создание тепловой карты дефектов
        """
        height, width = image.shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        for detection in detections:
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            
            x1 = int(bbox["x1"])
            y1 = int(bbox["y1"])
            x2 = int(bbox["x2"])
            y2 = int(bbox["y2"])
            
            # Добавляем "тепло" в область дефекта
            heatmap[y1:y2, x1:x2] += confidence
        
        # Нормализуем и конвертируем в цветное изображение
        heatmap = np.clip(heatmap, 0, 1)
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return heatmap_colored
    
    def _create_stats_panel(self, width: int, height: int, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Создание панели со статистикой
        """
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        
        # Заголовок
        cv2.putText(panel, "СТАТИСТИКА ДЕФЕКТОВ", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Общее количество
        total = len(detections)
        cv2.putText(panel, f"Всего найдено: {total}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # По типам
        defect_counts = {}
        for detection in detections:
            class_name = detection["class"]
            defect_counts[class_name] = defect_counts.get(class_name, 0) + 1
        
        y_offset = 90
        for class_name, count in defect_counts.items():
            ru_name = self.class_names_ru.get(class_name, class_name)
            color = self.colors.get(class_name, (255, 255, 255))
            text = f"{ru_name}: {count}"
            cv2.putText(panel, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 25
        
        return panel
