# utils/car_evaluator.py
import logging
from typing import Dict, List, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class CarEvaluator:
    """
    Система оценки состояния автомобиля и рекомендаций
    """
    
    def __init__(self):
        # Веса для разных типов дефектов
        self.defect_weights = {
            "rust": 0.4,      # Ржавчина - очень серьезно
            "crack": 0.3,     # Трещины - серьезно
            "dent": 0.2,      # Вмятины - средне
            "scratch": 0.1    # Царапины - менее серьезно
        }
        
        # Пороги для оценки
        self.thresholds = {
            "excellent": 0.9,    # Отличное состояние
            "good": 0.7,         # Хорошее состояние
            "fair": 0.5,         # Удовлетворительное
            "poor": 0.3,         # Плохое состояние
            "critical": 0.0      # Критическое состояние
        }
        
        # Рекомендации на основе оценки
        self.recommendations = {
            "excellent": {
                "buy": True,
                "confidence": "high",
                "message": "🚗 Отличное состояние! Рекомендуем к покупке.",
                "details": "Автомобиль в отличном состоянии с минимальными дефектами."
            },
            "good": {
                "buy": True,
                "confidence": "medium",
                "message": "✅ Хорошее состояние. Можно покупать с осторожностью.",
                "details": "Небольшие дефекты, но в целом автомобиль в хорошем состоянии."
            },
            "fair": {
                "buy": False,
                "confidence": "medium",
                "message": "⚠️ Удовлетворительное состояние. Требует внимания.",
                "details": "Обнаружены заметные дефекты. Рекомендуем тщательный осмотр."
            },
            "poor": {
                "buy": False,
                "confidence": "high",
                "message": "❌ Плохое состояние. Не рекомендуется к покупке.",
                "details": "Множественные серьезные дефекты. Высокие затраты на ремонт."
            },
            "critical": {
                "buy": False,
                "confidence": "high",
                "message": "🚨 Критическое состояние! Категорически не рекомендуется!",
                "details": "Критические дефекты. Автомобиль небезопасен для эксплуатации."
            }
        }
    
    def evaluate_car(self, detections: List[Dict[str, Any]], 
                    image_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        Оценка состояния автомобиля на основе обнаруженных дефектов
        
        Args:
            detections: Список обнаруженных дефектов
            image_size: Размер изображения (width, height)
            
        Returns:
            Словарь с оценкой и рекомендациями
        """
        try:
            # Базовые метрики
            total_defects = len(detections)
            image_area = image_size[0] * image_size[1]
            
            if total_defects == 0:
                return self._create_excellent_rating()
            
            # Анализируем дефекты
            defect_analysis = self._analyze_defects(detections, image_area)
            
            # Вычисляем общую оценку
            overall_score = self._calculate_overall_score(defect_analysis)
            
            # Определяем категорию состояния
            condition_category = self._categorize_condition(overall_score)
            
            # Получаем рекомендации
            recommendation = self.recommendations[condition_category]
            
            # Создаем детальный отчет
            evaluation = {
                "overall_score": round(overall_score, 2),
                "condition_category": condition_category,
                "recommendation": recommendation,
                "defect_analysis": defect_analysis,
                "summary": {
                    "total_defects": total_defects,
                    "critical_defects": defect_analysis["critical_count"],
                    "defect_coverage": round(defect_analysis["coverage_percentage"], 2),
                    "severity_index": round(defect_analysis["severity_index"], 2)
                }
            }
            
            logger.info(f"Оценка автомобиля: {condition_category} (score: {overall_score:.2f})")
            return evaluation
            
        except Exception as e:
            logger.error(f"Ошибка при оценке автомобиля: {str(e)}")
            return self._create_error_rating()
    
    def _analyze_defects(self, detections: List[Dict[str, Any]], 
                        image_area: float) -> Dict[str, Any]:
        """Анализ дефектов"""
        defect_counts = {}
        total_severity = 0.0
        total_area = 0.0
        critical_defects = 0
        
        for detection in detections:
            class_name = detection["class"]
            confidence = detection["confidence"]
            area = detection["area"]
            
            # Подсчитываем дефекты по типам
            defect_counts[class_name] = defect_counts.get(class_name, 0) + 1
            
            # Вычисляем серьезность дефекта
            weight = self.defect_weights.get(class_name, 0.1)
            severity = confidence * weight
            total_severity += severity
            
            # Площадь дефектов
            total_area += area
            
            # Критические дефекты (высокая уверенность + серьезный тип)
            if confidence > 0.7 and class_name in ["rust", "crack"]:
                critical_defects += 1
        
        # Процент покрытия дефектами
        coverage_percentage = (total_area / image_area) * 100
        
        # Индекс серьезности
        severity_index = total_severity / len(detections) if detections else 0
        
        return {
            "defect_counts": defect_counts,
            "total_severity": total_severity,
            "coverage_percentage": coverage_percentage,
            "severity_index": severity_index,
            "critical_count": critical_defects
        }
    
    def _calculate_overall_score(self, defect_analysis: Dict[str, Any]) -> float:
        """Вычисление общей оценки (0-1, где 1 = отлично)"""
        base_score = 1.0
        
        # Штрафы за дефекты
        severity_penalty = defect_analysis["severity_index"] * 0.4
        coverage_penalty = min(defect_analysis["coverage_percentage"] / 10, 0.3)
        critical_penalty = defect_analysis["critical_count"] * 0.2
        
        # Итоговая оценка
        final_score = base_score - severity_penalty - coverage_penalty - critical_penalty
        
        return max(0.0, min(1.0, final_score))
    
    def _categorize_condition(self, score: float) -> str:
        """Категоризация состояния по оценке"""
        if score >= self.thresholds["excellent"]:
            return "excellent"
        elif score >= self.thresholds["good"]:
            return "good"
        elif score >= self.thresholds["fair"]:
            return "fair"
        elif score >= self.thresholds["poor"]:
            return "poor"
        else:
            return "critical"
    
    def _create_excellent_rating(self) -> Dict[str, Any]:
        """Создание отличной оценки"""
        return {
            "overall_score": 1.0,
            "condition_category": "excellent",
            "recommendation": self.recommendations["excellent"],
            "defect_analysis": {
                "defect_counts": {},
                "total_severity": 0.0,
                "coverage_percentage": 0.0,
                "severity_index": 0.0,
                "critical_count": 0
            },
            "summary": {
                "total_defects": 0,
                "critical_defects": 0,
                "defect_coverage": 0.0,
                "severity_index": 0.0
            }
        }
    
    def _create_error_rating(self) -> Dict[str, Any]:
        """Создание оценки при ошибке"""
        return {
            "overall_score": 0.0,
            "condition_category": "critical",
            "recommendation": {
                "buy": False,
                "confidence": "low",
                "message": "❌ Ошибка анализа. Требуется повторная проверка.",
                "details": "Не удалось проанализировать изображение."
            },
            "defect_analysis": {
                "defect_counts": {},
                "total_severity": 0.0,
                "coverage_percentage": 0.0,
                "severity_index": 0.0,
                "critical_count": 0
            },
            "summary": {
                "total_defects": 0,
                "critical_defects": 0,
                "defect_coverage": 0.0,
                "severity_index": 0.0
            }
        }
    
    def get_detailed_advice(self, evaluation: Dict[str, Any]) -> str:
        """Получение детального совета"""
        category = evaluation["condition_category"]
        summary = evaluation["summary"]
        
        advice_parts = [evaluation["recommendation"]["message"]]
        
        if summary["total_defects"] > 0:
            advice_parts.append(f"\n📊 Статистика:")
            advice_parts.append(f"• Всего дефектов: {summary['total_defects']}")
            advice_parts.append(f"• Критических: {summary['critical_defects']}")
            advice_parts.append(f"• Покрытие дефектами: {summary['defect_coverage']}%")
            advice_parts.append(f"• Индекс серьезности: {summary['severity_index']:.2f}")
        
        if category in ["poor", "critical"]:
            advice_parts.append(f"\n💡 Рекомендации:")
            advice_parts.append("• Проведите профессиональный осмотр")
            advice_parts.append("• Оцените стоимость ремонта")
            advice_parts.append("• Рассмотрите другие варианты")
        elif category == "fair":
            advice_parts.append(f"\n💡 Рекомендации:")
            advice_parts.append("• Дополнительный осмотр механика")
            advice_parts.append("• Учет затрат на устранение дефектов")
        
        return "\n".join(advice_parts)


