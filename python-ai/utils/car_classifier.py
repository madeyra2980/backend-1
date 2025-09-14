# utils/car_classifier.py
import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class CleanlinessLevel(Enum):
    CLEAN = "clean"
    DIRTY = "dirty"

class IntegrityLevel(Enum):
    INTACT = "intact"
    DAMAGED = "damaged"

class CarClassifier:
    """
    Классификатор автомобилей по чистоте и целостности
    """
    
    def __init__(self):
        # Пороги для классификации
        self.cleanliness_threshold = 0.3  # Порог для определения чистоты
        self.integrity_threshold = 0.2    # Порог для определения целостности
        
        # Веса для разных типов дефектов при оценке целостности
        self.damage_weights = {
            "rust": 0.4,      # Ржавчина - серьезное повреждение
            "crack": 0.3,     # Трещины - серьезное повреждение
            "dent": 0.2,      # Вмятины - среднее повреждение
            "scratch": 0.1    # Царапины - легкое повреждение
        }
    
    def classify_car(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Классификация автомобиля по чистоте и целостности
        
        Args:
            image: Изображение автомобиля (BGR)
            detections: Список обнаруженных дефектов
            
        Returns:
            Словарь с классификацией
        """
        try:
            # Анализируем чистоту
            cleanliness = self._analyze_cleanliness(image)
            
            # Анализируем целостность
            integrity = self._analyze_integrity(detections, image.shape)
            
            # Определяем общую оценку
            overall_score = self._calculate_overall_score(cleanliness, integrity)
            
            # Генерируем рекомендацию
            recommendation = self._generate_recommendation(cleanliness, integrity, overall_score)
            
            return {
                "cleanliness": cleanliness,
                "integrity": integrity,
                "overall_score": overall_score,
                "recommendation": recommendation,
                "summary": {
                    "is_clean": cleanliness["level"] == CleanlinessLevel.CLEAN,
                    "is_intact": integrity["level"] == IntegrityLevel.INTACT,
                    "total_defects": len(detections),
                    "damage_severity": integrity["severity_score"]
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка классификации автомобиля: {str(e)}")
            return self._create_error_classification()
    
    def _analyze_cleanliness(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Анализ чистоты автомобиля
        """
        try:
            # Конвертируем в HSV для лучшего анализа цвета
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Анализируем яркость и контраст
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Вычисляем метрики чистоты
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Анализируем равномерность цвета (грязные машины имеют неравномерный цвет)
            color_variance = self._calculate_color_variance(image)
            
            # Анализируем наличие грязи (темные пятна)
            dirt_score = self._detect_dirt_spots(image)
            
            # Комбинированная оценка чистоты
            cleanliness_score = self._calculate_cleanliness_score(
                brightness, contrast, color_variance, dirt_score
            )
            
            level = CleanlinessLevel.CLEAN if cleanliness_score > self.cleanliness_threshold else CleanlinessLevel.DIRTY
            
            return {
                "level": level,
                "score": cleanliness_score,
                "brightness": brightness,
                "contrast": contrast,
                "color_variance": color_variance,
                "dirt_score": dirt_score,
                "description": self._get_cleanliness_description(level, cleanliness_score)
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа чистоты: {str(e)}")
            return {
                "level": CleanlinessLevel.DIRTY,
                "score": 0.0,
                "description": "Ошибка анализа чистоты"
            }
    
    def _analyze_integrity(self, detections: List[Dict[str, Any]], image_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Анализ целостности автомобиля
        """
        try:
            if not detections:
                return {
                    "level": IntegrityLevel.INTACT,
                    "score": 1.0,
                    "severity_score": 0.0,
                    "defect_count": 0,
                    "description": "Автомобиль в отличном состоянии, без видимых повреждений"
                }
            
            # Вычисляем общую серьезность повреждений
            total_severity = 0.0
            defect_counts = {}
            total_area = 0.0
            image_area = image_shape[0] * image_shape[1]
            
            for detection in detections:
                class_name = detection["class"]
                confidence = detection["confidence"]
                area = detection["area"]
                
                # Подсчитываем дефекты
                defect_counts[class_name] = defect_counts.get(class_name, 0) + 1
                
                # Вычисляем серьезность
                weight = self.damage_weights.get(class_name, 0.1)
                severity = confidence * weight
                total_severity += severity
                
                total_area += area
            
            # Нормализуем оценку
            severity_score = total_severity / len(detections) if detections else 0.0
            coverage_percentage = (total_area / image_area) * 100
            
            # Определяем уровень целостности
            integrity_score = max(0.0, 1.0 - severity_score - (coverage_percentage / 100) * 0.3)
            level = IntegrityLevel.INTACT if integrity_score > self.integrity_threshold else IntegrityLevel.DAMAGED
            
            return {
                "level": level,
                "score": integrity_score,
                "severity_score": severity_score,
                "defect_count": len(detections),
                "coverage_percentage": coverage_percentage,
                "defect_counts": defect_counts,
                "description": self._get_integrity_description(level, severity_score, len(detections))
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа целостности: {str(e)}")
            return {
                "level": IntegrityLevel.DAMAGED,
                "score": 0.0,
                "severity_score": 1.0,
                "description": "Ошибка анализа целостности"
            }
    
    def _calculate_color_variance(self, image: np.ndarray) -> float:
        """Вычисление вариации цвета"""
        # Разбиваем изображение на блоки и анализируем цветовую вариацию
        h, w = image.shape[:2]
        block_size = 32
        
        variances = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = image[y:y+block_size, x:x+block_size]
                if block.size > 0:
                    mean_color = np.mean(block, axis=(0, 1))
                    variance = np.var(block, axis=(0, 1))
                    variances.append(np.mean(variance))
        
        return np.mean(variances) if variances else 0.0
    
    def _detect_dirt_spots(self, image: np.ndarray) -> float:
        """Обнаружение грязных пятен"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Ищем темные области (потенциальная грязь)
        dark_threshold = np.percentile(gray, 20)  # Нижние 20% по яркости
        dark_mask = gray < dark_threshold
        
        # Фильтруем по размеру (убираем мелкие шумы)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark_mask = cv2.morphologyEx(dark_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        # Вычисляем процент темных областей
        dirt_percentage = np.sum(dark_mask) / dark_mask.size
        
        return min(1.0, dirt_percentage * 2)  # Нормализуем к 0-1
    
    def _calculate_cleanliness_score(self, brightness: float, contrast: float, 
                                   color_variance: float, dirt_score: float) -> float:
        """Вычисление оценки чистоты"""
        # Яркие и контрастные изображения обычно чище
        brightness_score = min(1.0, brightness / 128)  # Нормализуем к 0-1
        contrast_score = min(1.0, contrast / 64)       # Нормализуем к 0-1
        
        # Низкая вариация цвета и мало грязи = чистота
        variance_score = max(0.0, 1.0 - color_variance / 1000)
        dirt_penalty = 1.0 - dirt_score
        
        # Комбинированная оценка
        cleanliness_score = (
            brightness_score * 0.3 +
            contrast_score * 0.3 +
            variance_score * 0.2 +
            dirt_penalty * 0.2
        )
        
        return max(0.0, min(1.0, cleanliness_score))
    
    def _calculate_overall_score(self, cleanliness: Dict[str, Any], 
                               integrity: Dict[str, Any]) -> float:
        """Вычисление общей оценки"""
        cleanliness_weight = 0.3  # Чистота менее важна
        integrity_weight = 0.7    # Целостность более важна
        
        overall_score = (
            cleanliness["score"] * cleanliness_weight +
            integrity["score"] * integrity_weight
        )
        
        return max(0.0, min(1.0, overall_score))
    
    def _generate_recommendation(self, cleanliness: Dict[str, Any], 
                               integrity: Dict[str, Any], 
                               overall_score: float) -> Dict[str, Any]:
        """Генерация рекомендации"""
        is_clean = cleanliness["level"] == CleanlinessLevel.CLEAN
        is_intact = integrity["level"] == IntegrityLevel.INTACT
        
        # Сообщения для водителей
        driver_messages = {
            (True, True): {
                "message": "🚗 Вы хороший водитель!",
                "details": "Машина чистая + целая = все хорошо! Продолжайте в том же духе! 😊",
                "emoji": "😊"
            },
            (True, False): {
                "message": "🔧 Надо в ремонт!",
                "details": "Чистая + побитая = пора к механику! Но хотя бы мойте машину 👍",
                "emoji": "🔧"
            },
            (False, True): {
                "message": "🧽 Пора на мойку!",
                "details": "Целая + грязная = отличная машина, но нужна мойка! 💦",
                "emoji": "🧽"
            },
            (False, False): {
                "message": "😴 Лучше сегодня отдохнуть!",
                "details": "Грязная + побитая = лучше не ездить сегодня! Отдохните! 😄",
                "emoji": "😴"
            }
        }
        
        # Сообщения для пассажиров
        passenger_messages = {
            (True, True): {
                "message": "👑 Лучше только этой машины!",
                "details": "Чистая + целая = 'бывшая' машина! Садитесь смело! 😉",
                "emoji": "👑"
            },
            (True, False): {
                "message": "🐐 Без козлов на дороге не обойтись!",
                "details": "Чистая + побитая = осторожно с водителем! 🙃",
                "emoji": "🐐"
            },
            (False, True): {
                "message": "🌧️ Последнее время слишком много дождя!",
                "details": "Целая + грязная = хорошая машина, просто дождь был! 😗",
                "emoji": "🌧️"
            },
            (False, False): {
                "message": "💀 Хочешь жить? Садись к нам!",
                "details": "Грязная + побитая = экстремальная поездка! 😈",
                "emoji": "💀"
            }
        }
        
        # Выбираем сообщения на основе состояния
        driver_msg = driver_messages[(is_clean, is_intact)]
        passenger_msg = passenger_messages[(is_clean, is_intact)]
        
        return {
            "driver": driver_msg,
            "passenger": passenger_msg,
            "is_clean": is_clean,
            "is_intact": is_intact,
            "overall_condition": f"{'Чистая' if is_clean else 'Грязная'} + {'Целая' if is_intact else 'Побитая'}"
        }
    
    def _get_cleanliness_description(self, level: CleanlinessLevel, score: float) -> str:
        """Получение описания чистоты"""
        if level == CleanlinessLevel.CLEAN:
            if score > 0.8:
                return "Автомобиль в отличном состоянии, очень чистый"
            else:
                return "Автомобиль чистый, в хорошем состоянии"
        else:
            if score < 0.2:
                return "Автомобиль очень грязный, требует мойки"
            else:
                return "Автомобиль загрязнен, рекомендуется мойка"
    
    def _get_integrity_description(self, level: IntegrityLevel, severity: float, defect_count: int) -> str:
        """Получение описания целостности"""
        if level == IntegrityLevel.INTACT:
            return "Автомобиль без видимых повреждений, в отличном состоянии"
        else:
            if severity > 0.7:
                return f"Автомобиль серьезно поврежден ({defect_count} дефектов)"
            elif severity > 0.4:
                return f"Автомобиль имеет заметные повреждения ({defect_count} дефектов)"
            else:
                return f"Автомобиль имеет незначительные повреждения ({defect_count} дефектов)"
    
    def _create_error_classification(self) -> Dict[str, Any]:
        """Создание классификации при ошибке"""
        return {
            "cleanliness": {
                "level": CleanlinessLevel.DIRTY,
                "score": 0.0,
                "description": "Ошибка анализа чистоты"
            },
            "integrity": {
                "level": IntegrityLevel.DAMAGED,
                "score": 0.0,
                "description": "Ошибка анализа целостности"
            },
            "overall_score": 0.0,
            "recommendation": {
                "buy": False,
                "confidence": "low",
                "message": "❌ Ошибка анализа. Требуется повторная проверка.",
                "details": "Не удалось проанализировать изображение.",
                "category": "error"
            },
            "summary": {
                "is_clean": False,
                "is_intact": False,
                "total_defects": 0,
                "damage_severity": 1.0
            }
        }


