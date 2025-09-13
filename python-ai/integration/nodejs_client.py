# integration/nodejs_client.py
"""
Клиент для интеграции Python AI с Node.js API
"""
import requests
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class NodeJSClient:
    """
    Клиент для взаимодействия с Node.js API
    """
    
    def __init__(self, nodejs_url: str = "http://localhost:1015"):
        self.nodejs_url = nodejs_url
        self.session = requests.Session()
    
    def upload_file(self, file_path: str) -> Dict[str, Any]:
        """
        Загрузка файла в Node.js API
        """
        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                response = self.session.post(f"{self.nodejs_url}/api/uploads", files=files)
            
            if response.status_code == 200:
                return {"success": True, "data": response.content}
            else:
                return {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"Ошибка загрузки файла: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_files(self, page: int = 1, limit: int = 10) -> Dict[str, Any]:
        """
        Получение списка файлов
        """
        try:
            params = {"page": page, "limit": limit}
            response = self.session.get(f"{self.nodejs_url}/api/files", params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"Ошибка получения файлов: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_file_by_id(self, file_id: str) -> Dict[str, Any]:
        """
        Получение файла по ID
        """
        try:
            response = self.session.get(f"{self.nodejs_url}/api/files/{file_id}")
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"Ошибка получения файла: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def delete_file(self, file_id: str) -> Dict[str, Any]:
        """
        Удаление файла
        """
        try:
            response = self.session.delete(f"{self.nodejs_url}/api/files/{file_id}")
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"Ошибка удаления файла: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Проверка состояния Node.js API
        """
        try:
            response = self.session.get(f"{self.nodejs_url}/api/health")
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"Ошибка проверки здоровья: {str(e)}")
            return {"success": False, "error": str(e)}
