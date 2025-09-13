# utils/image_processor.py
import cv2
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Класс для предобработки изображений автомобилей
    """
    
    def __init__(self):
        self.target_size = (640, 640)  # Стандартный размер для YOLO
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Предобработка изображения для детекции
        
        Args:
            image: Входное изображение (BGR)
            
        Returns:
            Обработанное изображение
        """
        try:
            # Изменяем размер изображения
            resized = cv2.resize(image, self.target_size)
            
            # Улучшаем контраст
            enhanced = self.enhance_contrast(resized)
            
            # Убираем шум
            denoised = self.denoise(enhanced)
            
            logger.info("Изображение успешно предобработано")
            return denoised
            
        except Exception as e:
            logger.error(f"Ошибка предобработки: {str(e)}")
            return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Улучшение контраста изображения
        """
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Удаление шума с изображения
        """
        # Нелокальное среднее для удаления шума
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    def detect_car_region(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Обнаружение области автомобиля на изображении
        """
        # Простой метод обнаружения автомобиля по цвету и форме
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Ищем контуры
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Находим самый большой контур (предположительно автомобиль)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return x, y, x + w, y + h
        
        # Если не нашли, возвращаем весь кадр
        return 0, 0, image.shape[1], image.shape[0]
    
    def crop_car_region(self, image: np.ndarray) -> np.ndarray:
        """
        Обрезка области автомобиля
        """
        x1, y1, x2, y2 = self.detect_car_region(image)
        return image[y1:y2, x1:x2]
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Нормализация изображения
        """
        # Нормализация к диапазону [0, 1]
        normalized = image.astype(np.float32) / 255.0
        return normalized
