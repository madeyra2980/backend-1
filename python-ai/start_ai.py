#!/usr/bin/env python3
"""
Скрипт запуска Python AI сервиса
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Проверка установленных зависимостей"""
    try:
        import torch
        import cv2
        import fastapi
        import ultralytics
        logger.info("✅ Все зависимости установлены")
        return True
    except ImportError as e:
        logger.error(f"❌ Отсутствует зависимость: {e}")
        return False

def install_dependencies():
    """Установка зависимостей"""
    logger.info("📦 Устанавливаем зависимости...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        logger.info("✅ Зависимости установлены")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Ошибка установки: {e}")
        return False

def create_directories():
    """Создание необходимых директорий"""
    dirs = [
        "data/images/train",
        "data/images/val", 
        "data/images/test",
        "data/labels/train",
        "data/labels/val",
        "data/labels/test",
        "models",
        "runs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("📁 Директории созданы")

def start_server():
    """Запуск FastAPI сервера"""
    logger.info("🚀 Запускаем Python AI сервер...")
    try:
        subprocess.run([sys.executable, "api/main.py"])
    except KeyboardInterrupt:
        logger.info("👋 Сервер остановлен")
    except Exception as e:
        logger.error(f"❌ Ошибка запуска: {e}")

def main():
    """Главная функция"""
    logger.info("🤖 Car Defect Detection AI - Запуск")
    
    # Проверяем зависимости
    if not check_dependencies():
        logger.info("📦 Устанавливаем недостающие зависимости...")
        if not install_dependencies():
            logger.error("❌ Не удалось установить зависимости")
            return
    
    # Создаем директории
    create_directories()
    
    # Запускаем сервер
    start_server()

if __name__ == "__main__":
    main()
