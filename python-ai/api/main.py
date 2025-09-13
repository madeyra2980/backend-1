# api/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io
import cv2
import numpy as np
from PIL import Image
import logging
from typing import List, Dict, Any
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detector import CarDefectDetector
from utils.image_processor import ImageProcessor
from utils.visualizer import DetectionVisualizer
from utils.car_classifier import CarClassifier

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Car Defect Detection API",
    description="API для обнаружения ржавчины и царапин на автомобилях",
    version="1.0.0"
)

# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1015", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация компонентов
detector = CarDefectDetector()
image_processor = ImageProcessor()
visualizer = DetectionVisualizer()
classifier = CarClassifier()

@app.get("/")
async def root():
    return {
        "message": "Car Defect Detection API",
        "version": "1.0.0",
        "endpoints": {
            "detect": "POST /detect",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": detector.is_model_loaded(),
        "timestamp": "2025-09-13T10:50:00Z"
    }

@app.post("/detect")
async def detect_defects(file: UploadFile = File(...)):
    """
    Обнаружение дефектов на изображении автомобиля
    """
    try:
        # Проверяем тип файла
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Файл должен быть изображением")
        
        # Читаем изображение
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        logger.info(f"Обрабатываем изображение: {file.filename}")
        
        # Предобработка изображения
        processed_image = image_processor.preprocess(image_cv)
        
        # Детекция дефектов
        detections = detector.detect(processed_image)
        
        # Визуализация результатов
        result_image = visualizer.draw_detections(
            image_cv, 
            detections, 
            confidence_threshold=0.3
        )
        
        # Классификация автомобиля по чистоте и целостности
        classification = classifier.classify_car(image_cv, detections)
        
        # Подготавливаем ответ
        response_data = {
            "success": True,
            "filename": file.filename,
            "detections": detections,
            "total_defects": len(detections),
            "defect_types": list(set([d["class"] for d in detections])),
            "classification": classification,
            "image_size": {
                "width": image_cv.shape[1],
                "height": image_cv.shape[0]
            }
        }
        
        # Конвертируем изображение в байты
        _, buffer = cv2.imencode('.jpg', result_image)
        image_bytes = io.BytesIO(buffer).getvalue()
        
        # Возвращаем изображение с разметкой
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/jpeg",
            headers={
                "X-Detection-Data": json.dumps(response_data, ensure_ascii=False)
            }
        )
        
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.post("/detect-json")
async def detect_defects_json(file: UploadFile = File(...)):
    """
    Обнаружение дефектов с возвратом JSON данных
    """
    try:
        # Читаем изображение
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Предобработка
        processed_image = image_processor.preprocess(image_cv)
        
        # Детекция
        detections = detector.detect(processed_image)
        
        # Классификация автомобиля по чистоте и целостности
        classification = classifier.classify_car(image_cv, detections)
        
        return {
            "success": True,
            "filename": file.filename,
            "detections": detections,
            "total_defects": len(detections),
            "defect_types": list(set([d["class"] for d in detections])),
            "image_size": {
                "width": image_cv.shape[1],
                "height": image_cv.shape[0]
            },
            "classification": classification
        }
        
    except Exception as e:
        logger.error(f"Ошибка при обработке: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
