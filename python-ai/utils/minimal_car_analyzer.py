# utils/minimal_car_analyzer.py
import cv2
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MinimalCarAnalyzer:
    """
    Минимальный анализатор автомобилей
    """
    
    def __init__(self):
        pass
    
    def analyze_car_condition(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Минимальный анализ состояния автомобиля
        """
        try:
            # Простой анализ изображения
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Анализ яркости и контраста
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Простой анализ "грязи" (темные области)
            dark_threshold = np.percentile(gray, 15)
            dark_pixels = np.sum(gray < dark_threshold)
            total_pixels = image.shape[0] * image.shape[1]
            dirt_percentage = (dark_pixels / total_pixels) * 100
            
            # Простой анализ "повреждений" (изменения интенсивности)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            damage_pixels = np.sum(laplacian > 30)
            damage_percentage = (damage_pixels / total_pixels) * 100
            
            # Оценка чистоты
            cleanliness_score = max(0, 1 - (dirt_percentage / 15))  # 15% грязи = 0 баллов
            
            # Оценка целостности
            integrity_score = max(0, 1 - (damage_percentage / 10))  # 10% повреждений = 0 баллов
            
            # Общая оценка
            overall_score = (cleanliness_score * 0.3 + integrity_score * 0.7)
            
            # Определяем состояние
            if overall_score >= 0.8:
                condition = "excellent"
                condition_name = "Отличное"
            elif overall_score >= 0.6:
                condition = "good"
                condition_name = "Хорошее"
            elif overall_score >= 0.4:
                condition = "fair"
                condition_name = "Удовлетворительное"
            elif overall_score >= 0.2:
                condition = "poor"
                condition_name = "Плохое"
            else:
                condition = "critical"
                condition_name = "Критическое"
            
            # Генерируем рекомендации
            recommendation = self._generate_recommendation(condition, overall_score)
            
            return {
                "rust": {
                    "detected": False,
                    "percentage": 0.0,
                    "severity": 0.0,
                    "description": "Ржавчина не обнаружена"
                },
                "scratches": {
                    "detected": damage_percentage > 2,
                    "percentage": damage_percentage,
                    "severity": min(1.0, damage_percentage / 5),
                    "description": f"Царапины и повреждения: {damage_percentage:.1f}% поверхности"
                },
                "dents": {
                    "detected": damage_percentage > 3,
                    "percentage": damage_percentage,
                    "severity": min(1.0, damage_percentage / 8),
                    "description": f"Вмятины и деформации: {damage_percentage:.1f}% поверхности"
                },
                "cracks": {
                    "detected": damage_percentage > 1,
                    "percentage": damage_percentage,
                    "severity": min(1.0, damage_percentage / 3),
                    "description": f"Трещины и сколы: {damage_percentage:.1f}% поверхности"
                },
                "cleanliness": {
                    "score": cleanliness_score,
                    "brightness": brightness,
                    "contrast": contrast,
                    "dirt_percentage": dirt_percentage,
                    "is_clean": cleanliness_score > 0.7,
                    "description": f"Чистота: {cleanliness_score:.1%} (грязь: {dirt_percentage:.1f}%)"
                },
                "overall_condition": {
                    "score": overall_score,
                    "condition": condition,
                    "condition_name": condition_name,
                    "damage_score": 1 - integrity_score,
                    "cleanliness_score": cleanliness_score,
                    "description": f"{condition_name} состояние! Оценка: {overall_score:.1%}"
                },
                "recommendation": recommendation
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа автомобиля: {str(e)}")
            return self._create_error_analysis()
    
    def _generate_recommendation(self, condition: str, overall_score: float) -> Dict[str, Any]:
        """Генерация рекомендации"""
        
        # Сообщения для водителей
        driver_messages = {
            "excellent": {
                "message": "🚗 Отличная машина!",
                "details": "Машина в отличном состоянии! Продолжайте ухаживать за ней! 😊",
                "emoji": "😊"
            },
            "good": {
                "message": "✅ Хорошая машина!",
                "details": "Машина в хорошем состоянии, но есть небольшие дефекты. Стоит подумать о ремонте! 👍",
                "emoji": "👍"
            },
            "fair": {
                "message": "⚠️ Нужен ремонт!",
                "details": "Машина требует внимания! Рекомендуем показать механику! 🔧",
                "emoji": "🔧"
            },
            "poor": {
                "message": "❌ Серьезные проблемы!",
                "details": "Машина в плохом состоянии! Нужен срочный ремонт! 🚨",
                "emoji": "🚨"
            },
            "critical": {
                "message": "💀 Критическое состояние!",
                "details": "Машина в критическом состоянии! Лучше не ездить! 😱",
                "emoji": "😱"
            }
        }
        
        # Сообщения для пассажиров
        passenger_messages = {
            "excellent": {
                "message": "👑 Премиум класс!",
                "details": "Отличная машина! Садитесь смело! 😉",
                "emoji": "👑"
            },
            "good": {
                "message": "✅ Хороший выбор!",
                "details": "Неплохая машина, можно ехать! 😊",
                "emoji": "😊"
            },
            "fair": {
                "message": "⚠️ Осторожно!",
                "details": "Машина не в лучшем состоянии, но доедете! 🙃",
                "emoji": "🙃"
            },
            "poor": {
                "message": "🐐 Рискованно!",
                "details": "Машина в плохом состоянии! Будьте осторожны! 😬",
                "emoji": "😬"
            },
            "critical": {
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
            "overall_condition": condition,
            "score": overall_score
        }
    
    def _create_error_analysis(self) -> Dict[str, Any]:
        """Создание анализа при ошибке"""
        return {
            "rust": {"detected": False, "percentage": 0, "severity": 0, "description": "Ошибка анализа"},
            "scratches": {"detected": False, "percentage": 0, "severity": 0, "description": "Ошибка анализа"},
            "dents": {"detected": False, "percentage": 0, "severity": 0, "description": "Ошибка анализа"},
            "cracks": {"detected": False, "percentage": 0, "severity": 0, "description": "Ошибка анализа"},
            "cleanliness": {"score": 0, "is_clean": False, "description": "Ошибка анализа"},
            "overall_condition": {"score": 0, "condition": "critical", "description": "Ошибка анализа"},
            "recommendation": {
                "driver": {"message": "❌ Ошибка анализа", "details": "Не удалось проанализировать изображение", "emoji": "❌"},
                "passenger": {"message": "❌ Ошибка анализа", "details": "Не удалось проанализировать изображение", "emoji": "❌"}
            }
        }
