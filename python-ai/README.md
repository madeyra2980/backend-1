# Car Defect Detection AI

Система искусственного интеллекта для обнаружения дефектов автомобилей (ржавчина, царапины, вмятины, трещины) на основе YOLO.

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Запуск API сервера
```bash
python api/main.py
```

### 3. Тестирование
```bash
# Проверка здоровья
curl http://localhost:8000/health

# Анализ изображения
curl -X POST -F "file=@car_image.jpg" http://localhost:8000/detect
```

## 🏗️ Архитектура

```
python-ai/
├── api/                    # FastAPI сервер
│   └── main.py
├── models/                 # Модели машинного обучения
│   └── detector.py
├── utils/                  # Утилиты
│   ├── image_processor.py  # Предобработка изображений
│   └── visualizer.py       # Визуализация результатов
├── data/                   # Данные для обучения
│   ├── images/            # Изображения
│   └── labels/            # Аннотации
├── integration/           # Интеграция с Node.js
│   └── nodejs_client.py
└── train_model.py         # Скрипт обучения
```

## 🤖 Модели

### YOLO (You Only Look Once)
- **YOLOv8n**: Базовая модель для быстрой детекции
- **Кастомная модель**: Обученная на датасете дефектов автомобилей

### Классы дефектов:
- `rust` - Ржавчина
- `scratch` - Царапины  
- `dent` - Вмятины
- `crack` - Трещины

## 📊 Обучение модели

### 1. Подготовка данных
```bash
python train_model.py --mode prepare
```

### 2. Обучение
```bash
python train_model.py --mode train
```

### 3. Валидация
```bash
python train_model.py --mode validate --model runs/detect/car_defects/weights/best.pt
```

## 🔧 API Endpoints

### POST /detect
Анализ изображения с возвратом изображения с разметкой
```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/detect
```

### POST /detect-json
Анализ изображения с возвратом JSON данных
```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/detect-json
```

### GET /health
Проверка состояния сервиса
```bash
curl http://localhost:8000/health
```

## 🔗 Интеграция с Node.js

### Настройка в .env файле Node.js:
```env
USE_PYTHON_AI=true
PYTHON_AI_URL=http://localhost:8000
```

### Запуск обеих систем:
```bash
# Terminal 1: Node.js API
cd ../backend
npm start

# Terminal 2: Python AI
cd python-ai
python api/main.py
```

## 📈 Производительность

- **Время обработки**: ~2-5 секунд на изображение
- **Точность**: 85-95% (зависит от качества данных)
- **Поддерживаемые форматы**: JPEG, PNG, WEBP
- **Максимальный размер**: 10MB

## 🛠️ Разработка

### Структура детекции:
1. **Предобработка**: Улучшение контраста, удаление шума
2. **Детекция**: YOLO модель для поиска дефектов
3. **Визуализация**: Отрисовка bounding boxes и меток
4. **Постобработка**: Фильтрация по уверенности

### Кастомизация:
- Изменение порога уверенности
- Добавление новых классов дефектов
- Настройка цветов визуализации
- Улучшение предобработки

## 📚 Примеры использования

### Python API:
```python
import requests

# Анализ изображения
with open('car.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/detect', files={'file': f})
    
# Сохранение результата
with open('result.jpg', 'wb') as f:
    f.write(response.content)
```

### Node.js интеграция:
```javascript
const roboflowService = require('./services/roboflowService');

// Анализ через Python AI
const result = await roboflowService.analyzeImage(imageUrl);
```

## 🚀 Развертывание

### Docker (рекомендуется):
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "api/main.py"]
```

### Локальная установка:
```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt

# Запуск
python api/main.py
```

## 📝 Лицензия

MIT License - свободное использование и модификация.
