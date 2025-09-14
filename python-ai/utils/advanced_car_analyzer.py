# utils/advanced_car_analyzer.py
import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class CarCondition(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

class AdvancedCarAnalyzer:
    """
    Продвинутый анализатор автомобилей с использованием компьютерного зрения
    """
    
    def __init__(self):
        # Пороги для анализа
        self.damage_thresholds = {
            "rust": 0.3,        # Порог для ржавчины
            "scratches": 0.2,   # Порог для царапин
            "dents": 0.4,       # Порог для вмятин
            "cracks": 0.3       # Порог для трещин
        }
        
        # Веса для разных типов повреждений
        self.damage_weights = {
            "rust": 0.4,        # Ржавчина - очень серьезно
            "cracks": 0.35,     # Трещины - серьезно
            "dents": 0.2,       # Вмятины - средне
            "scratches": 0.05   # Царапины - менее серьезно
        }
    
    def analyze_car_condition(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Комплексный анализ состояния автомобиля
        """
        try:
            # Анализируем различные аспекты
            rust_analysis = self._detect_rust(image)
            scratch_analysis = self._detect_scratches(image)
            dent_analysis = self._detect_dents(image)
            crack_analysis = self._detect_cracks(image)
            cleanliness_analysis = self._analyze_cleanliness(image)
            
            # Объединяем результаты
            overall_condition = self._calculate_overall_condition(
                rust_analysis, scratch_analysis, dent_analysis, 
                crack_analysis, cleanliness_analysis
            )
            
            return {
                "rust": rust_analysis,
                "scratches": scratch_analysis,
                "dents": dent_analysis,
                "cracks": crack_analysis,
                "cleanliness": cleanliness_analysis,
                "overall_condition": overall_condition,
                "recommendation": self._generate_recommendation(overall_condition)
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа автомобиля: {str(e)}")
            return self._create_error_analysis()
    
    def _detect_rust(self, image: np.ndarray) -> Dict[str, Any]:
        """Детекция ржавчины"""
        try:
            # Простой анализ коричнево-оранжевых тонов
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Определяем диапазон цветов ржавчины
            lower_rust = np.array([0, 30, 30])
            upper_rust = np.array([25, 255, 255])
            
            # Создаем маску для ржавчины
            rust_mask = cv2.inRange(hsv, lower_rust, upper_rust)
            
            # Вычисляем процент ржавчины
            rust_pixels = np.sum(rust_mask > 0)
            total_pixels = image.shape[0] * image.shape[1]
            rust_percentage = (rust_pixels / total_pixels) * 100
            
            # Оценка серьезности ржавчины
            severity = min(1.0, rust_percentage / 5)  # 5% = максимальная серьезность
            
            return {
                "detected": rust_percentage > 0.3,
                "percentage": rust_percentage,
                "severity": severity,
                "areas_count": 1 if rust_percentage > 0.3 else 0,
                "total_area": rust_pixels,
                "description": self._get_rust_description(rust_percentage, 1 if rust_percentage > 0.3 else 0)
            }
            
        except Exception as e:
            logger.error(f"Ошибка детекции ржавчины: {str(e)}")
            return {"detected": False, "percentage": 0, "severity": 0, "description": "Ошибка анализа"}
    
    def _detect_scratches(self, image: np.ndarray) -> Dict[str, Any]:
        """Детекция царапин"""
        try:
            # Простой анализ краев
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Применяем детектор краев Канни
            edges = cv2.Canny(gray, 50, 150)
            
            # Вычисляем процент краев (потенциальные царапины)
            edge_pixels = np.sum(edges > 0)
            total_pixels = image.shape[0] * image.shape[1]
            scratch_percentage = (edge_pixels / total_pixels) * 100
            
            severity = min(1.0, scratch_percentage / 3)  # 3% = максимальная серьезность
            
            return {
                "detected": scratch_percentage > 0.5,
                "percentage": scratch_percentage,
                "severity": severity,
                "areas_count": 1 if scratch_percentage > 0.5 else 0,
                "total_area": edge_pixels,
                "description": self._get_scratch_description(scratch_percentage, 1 if scratch_percentage > 0.5 else 0)
            }
            
        except Exception as e:
            logger.error(f"Ошибка детекции царапин: {str(e)}")
            return {"detected": False, "percentage": 0, "severity": 0, "description": "Ошибка анализа"}
    
    def _detect_dents(self, image: np.ndarray) -> Dict[str, Any]:
        """Детекция вмятин"""
        try:
            # Простой анализ изменений интенсивности
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Применяем оператор Лапласа
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            
            # Пороговая обработка
            _, dent_mask = cv2.threshold(laplacian, 20, 255, cv2.THRESH_BINARY)
            
            # Вычисляем процент вмятин
            dent_pixels = np.sum(dent_mask > 0)
            total_pixels = image.shape[0] * image.shape[1]
            dent_percentage = (dent_pixels / total_pixels) * 100
            
            severity = min(1.0, dent_percentage / 5)  # 5% = максимальная серьезность
            
            return {
                "detected": dent_percentage > 0.5,
                "percentage": dent_percentage,
                "severity": severity,
                "areas_count": 1 if dent_percentage > 0.5 else 0,
                "total_area": dent_pixels,
                "description": self._get_dent_description(dent_percentage, 1 if dent_percentage > 0.5 else 0)
            }
            
        except Exception as e:
            logger.error(f"Ошибка детекции вмятин: {str(e)}")
            return {"detected": False, "percentage": 0, "severity": 0, "description": "Ошибка анализа"}
    
    def _detect_cracks(self, image: np.ndarray) -> Dict[str, Any]:
        """Детекция трещин"""
        try:
            # Простой анализ трещин
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Применяем детектор краев Канни
            edges = cv2.Canny(gray, 30, 100)
            
            # Вычисляем процент трещин
            crack_pixels = np.sum(edges > 0)
            total_pixels = image.shape[0] * image.shape[1]
            crack_percentage = (crack_pixels / total_pixels) * 100
            
            severity = min(1.0, crack_percentage / 2)  # 2% = максимальная серьезность
            
            return {
                "detected": crack_percentage > 0.3,
                "percentage": crack_percentage,
                "severity": severity,
                "areas_count": 1 if crack_percentage > 0.3 else 0,
                "total_area": crack_pixels,
                "description": self._get_crack_description(crack_percentage, 1 if crack_percentage > 0.3 else 0)
            }
            
        except Exception as e:
            logger.error(f"Ошибка детекции трещин: {str(e)}")
            return {"detected": False, "percentage": 0, "severity": 0, "description": "Ошибка анализа"}
    
    def _analyze_cleanliness(self, image: np.ndarray) -> Dict[str, Any]:
        """Анализ чистоты"""
        try:
            # Конвертируем в HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Анализируем яркость и контраст
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Ищем темные области (грязь)
            dark_threshold = np.percentile(gray, 20)
            dark_mask = gray < dark_threshold
            
            # Фильтруем по размеру
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dark_mask = cv2.morphologyEx(dark_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            
            # Вычисляем процент грязи
            dirt_pixels = np.sum(dark_mask > 0)
            total_pixels = image.shape[0] * image.shape[1]
            dirt_percentage = (dirt_pixels / total_pixels) * 100
            
            # Оценка чистоты
            cleanliness_score = max(0, 1 - (dirt_percentage / 20))  # 20% грязи = 0 баллов
            
            return {
                "score": cleanliness_score,
                "brightness": brightness,
                "contrast": contrast,
                "dirt_percentage": dirt_percentage,
                "is_clean": cleanliness_score > 0.7,
                "description": self._get_cleanliness_description(cleanliness_score, dirt_percentage)
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа чистоты: {str(e)}")
            return {"score": 0, "is_clean": False, "description": "Ошибка анализа"}
    
    def _calculate_overall_condition(self, rust, scratches, dents, cracks, cleanliness) -> Dict[str, Any]:
        """Вычисление общего состояния"""
        try:
            # Вычисляем общий балл повреждений
            damage_score = (
                rust["severity"] * self.damage_weights["rust"] +
                scratches["severity"] * self.damage_weights["scratches"] +
                dents["severity"] * self.damage_weights["dents"] +
                cracks["severity"] * self.damage_weights["cracks"]
            )
            
            # Учитываем чистоту
            cleanliness_weight = 0.2
            damage_weight = 0.8
            
            overall_score = max(0, 1 - damage_score * damage_weight + cleanliness["score"] * cleanliness_weight)
            
            # Определяем категорию состояния
            if overall_score >= 0.9:
                condition = CarCondition.EXCELLENT
            elif overall_score >= 0.7:
                condition = CarCondition.GOOD
            elif overall_score >= 0.5:
                condition = CarCondition.FAIR
            elif overall_score >= 0.3:
                condition = CarCondition.POOR
            else:
                condition = CarCondition.CRITICAL
            
            return {
                "score": overall_score,
                "condition": condition,
                "damage_score": damage_score,
                "cleanliness_score": cleanliness["score"],
                "description": self._get_condition_description(condition, overall_score)
            }
            
        except Exception as e:
            logger.error(f"Ошибка вычисления общего состояния: {str(e)}")
            return {"score": 0, "condition": CarCondition.CRITICAL, "description": "Ошибка анализа"}
    
    def _generate_recommendation(self, overall_condition: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация рекомендации"""
        condition = overall_condition["condition"]
        score = overall_condition["score"]
        
        # Сообщения для водителей
        driver_messages = {
            CarCondition.EXCELLENT: {
                "message": "🚗 Отличная машина!",
                "details": "Машина в отличном состоянии! Продолжайте ухаживать за ней! 😊",
                "emoji": "😊"
            },
            CarCondition.GOOD: {
                "message": "✅ Хорошая машина!",
                "details": "Машина в хорошем состоянии, но есть небольшие дефекты. Стоит подумать о ремонте! 👍",
                "emoji": "👍"
            },
            CarCondition.FAIR: {
                "message": "⚠️ Нужен ремонт!",
                "details": "Машина требует внимания! Рекомендуем показать механику! 🔧",
                "emoji": "🔧"
            },
            CarCondition.POOR: {
                "message": "❌ Серьезные проблемы!",
                "details": "Машина в плохом состоянии! Нужен срочный ремонт! 🚨",
                "emoji": "🚨"
            },
            CarCondition.CRITICAL: {
                "message": "💀 Критическое состояние!",
                "details": "Машина в критическом состоянии! Лучше не ездить! 😱",
                "emoji": "😱"
            }
        }
        
        # Сообщения для пассажиров
        passenger_messages = {
            CarCondition.EXCELLENT: {
                "message": "👑 Премиум класс!",
                "details": "Отличная машина! Садитесь смело! 😉",
                "emoji": "👑"
            },
            CarCondition.GOOD: {
                "message": "✅ Хороший выбор!",
                "details": "Неплохая машина, можно ехать! 😊",
                "emoji": "😊"
            },
            CarCondition.FAIR: {
                "message": "⚠️ Осторожно!",
                "details": "Машина не в лучшем состоянии, но доедете! 🙃",
                "emoji": "🙃"
            },
            CarCondition.POOR: {
                "message": "🐐 Рискованно!",
                "details": "Машина в плохом состоянии! Будьте осторожны! 😬",
                "emoji": "😬"
            },
            CarCondition.CRITICAL: {
                "message": "💀 Опасно для жизни!",
                "details": "Лучше не садиться! Машина в критическом состоянии! 😱",
                "emoji": "😱"
            }
        }
        
        driver_msg = driver_messages[condition]
        passenger_msg = passenger_messages[condition]
        
        return {
            "driver": driver_msg,
            "passenger": passenger_msg,
            "overall_condition": condition.value,
            "score": score
        }
    
    def _get_rust_description(self, percentage: float, areas_count: int) -> str:
        if percentage > 5:
            return f"Критическая ржавчина! {percentage:.1f}% поверхности поражено ({areas_count} участков)"
        elif percentage > 2:
            return f"Серьезная ржавчина! {percentage:.1f}% поверхности поражено ({areas_count} участков)"
        elif percentage > 0.5:
            return f"Заметная ржавчина! {percentage:.1f}% поверхности поражено ({areas_count} участков)"
        else:
            return f"Минимальная ржавчина: {percentage:.1f}% поверхности"
    
    def _get_scratch_description(self, percentage: float, areas_count: int) -> str:
        if percentage > 2:
            return f"Множественные царапины! {percentage:.1f}% поверхности ({areas_count} царапин)"
        elif percentage > 0.5:
            return f"Заметные царапины! {percentage:.1f}% поверхности ({areas_count} царапин)"
        else:
            return f"Незначительные царапины: {percentage:.1f}% поверхности"
    
    def _get_dent_description(self, percentage: float, areas_count: int) -> str:
        if percentage > 3:
            return f"Множественные вмятины! {percentage:.1f}% поверхности ({areas_count} вмятин)"
        elif percentage > 1:
            return f"Заметные вмятины! {percentage:.1f}% поверхности ({areas_count} вмятин)"
        else:
            return f"Незначительные вмятины: {percentage:.1f}% поверхности"
    
    def _get_crack_description(self, percentage: float, areas_count: int) -> str:
        if percentage > 1:
            return f"Критические трещины! {percentage:.1f}% поверхности ({areas_count} трещин)"
        elif percentage > 0.3:
            return f"Серьезные трещины! {percentage:.1f}% поверхности ({areas_count} трещин)"
        else:
            return f"Незначительные трещины: {percentage:.1f}% поверхности"
    
    def _get_cleanliness_description(self, score: float, dirt_percentage: float) -> str:
        if score > 0.8:
            return f"Отличная чистота! Машина очень чистая"
        elif score > 0.6:
            return f"Хорошая чистота! Небольшие загрязнения: {dirt_percentage:.1f}%"
        elif score > 0.4:
            return f"Удовлетворительная чистота! Заметные загрязнения: {dirt_percentage:.1f}%"
        else:
            return f"Плохая чистота! Машина очень грязная: {dirt_percentage:.1f}%"
    
    def _get_condition_description(self, condition: CarCondition, score: float) -> str:
        descriptions = {
            CarCondition.EXCELLENT: f"Отличное состояние! Оценка: {score:.1%}",
            CarCondition.GOOD: f"Хорошее состояние! Оценка: {score:.1%}",
            CarCondition.FAIR: f"Удовлетворительное состояние! Оценка: {score:.1%}",
            CarCondition.POOR: f"Плохое состояние! Оценка: {score:.1%}",
            CarCondition.CRITICAL: f"Критическое состояние! Оценка: {score:.1%}"
        }
        return descriptions[condition]
    
    def _create_error_analysis(self) -> Dict[str, Any]:
        """Создание анализа при ошибке"""
        return {
            "rust": {"detected": False, "percentage": 0, "severity": 0, "description": "Ошибка анализа"},
            "scratches": {"detected": False, "percentage": 0, "severity": 0, "description": "Ошибка анализа"},
            "dents": {"detected": False, "percentage": 0, "severity": 0, "description": "Ошибка анализа"},
            "cracks": {"detected": False, "percentage": 0, "severity": 0, "description": "Ошибка анализа"},
            "cleanliness": {"score": 0, "is_clean": False, "description": "Ошибка анализа"},
            "overall_condition": {"score": 0, "condition": CarCondition.CRITICAL, "description": "Ошибка анализа"},
            "recommendation": {
                "driver": {"message": "❌ Ошибка анализа", "details": "Не удалось проанализировать изображение", "emoji": "❌"},
                "passenger": {"message": "❌ Ошибка анализа", "details": "Не удалось проанализировать изображение", "emoji": "❌"}
            }
        }
