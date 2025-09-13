// services/roboflowService.js
const axios = require('axios');
const FormData = require('form-data');

class RoboflowService {
  constructor() {
    this.apiKey = process.env.ROBOFLOW_API_KEY;
    this.baseUrl = 'https://detect.roboflow.com';
    this.projectId = process.env.ROBOFLOW_PROJECT_ID || 'rust-and-scrach-cbhdg';
    this.version = process.env.ROBOFLOW_VERSION || '1';
    
    // Настройки Python AI сервиса
    this.pythonAIUrl = process.env.PYTHON_AI_URL || 'http://localhost:8000';
    this.usePythonAI = process.env.USE_PYTHON_AI === 'true';
    
    // Проверяем наличие API ключа
    if (!this.apiKey) {
        console.warn('⚠️ ROBOFLOW_API_KEY не установлен в переменных окружения');
    }
    
    if (this.usePythonAI) {
      console.log('🤖 Используется Python AI сервис:', this.pythonAIUrl);
    }
  }

  async analyzeImage(imageUrl) {
    try {
      // Если включен Python AI, используем его
      if (this.usePythonAI) {
        return await this.analyzeWithPythonAI(imageUrl);
      }
      
      // Иначе используем Roboflow
      return await this.analyzeWithRoboflow(imageUrl);
      
    } catch (error) {
      console.error('❌ Ошибка анализа изображения:', error.message);
      throw error;
    }
  }

  async analyzeWithPythonAI(imageUrl) {
    try {
      console.log('🤖 Анализируем изображение через Python AI...');
      
      // Скачиваем изображение
      const imageResponse = await axios.get(imageUrl, { responseType: 'arraybuffer' });
      
      // Отправляем в Python AI сервис
      const formData = new FormData();
      formData.append('file', imageResponse.data, {
        filename: 'image.jpg',
        contentType: 'image/jpeg'
      });
      
      const response = await axios({
        method: 'POST',
        url: `${this.pythonAIUrl}/detect`,
        data: formData,
        headers: {
          ...formData.getHeaders()
        },
        responseType: 'arraybuffer',
        timeout: 30000
      });

      console.log('✅ Python AI анализ завершен успешно');
      
      // Извлекаем данные классификации из заголовков
      const detectionData = response.headers['x-detection-data'];
      let classificationData = null;
      
      if (detectionData) {
        try {
          classificationData = JSON.parse(detectionData);
        } catch (e) {
          console.warn('Не удалось распарсить данные классификации:', e.message);
        }
      }
      
      return {
        success: true,
        data: response.data,
        contentType: 'image/jpeg',
        classification: classificationData
      };
    } catch (error) {
      console.error('❌ Ошибка Python AI:', error.message);
      throw new Error(`Python AI Error: ${error.message}`);
    }
  }

  async analyzeWithRoboflow(imageUrl) {
    try {
      // Проверяем наличие API ключа
      if (!this.apiKey) {
        throw new Error('ROBOFLOW_API_KEY не установлен. Добавьте его в .env файл');
      }
      
      console.log('🔍 Анализируем изображение через Roboflow...');
      console.log('🔑 Используем API ключ:', this.apiKey.substring(0, 8) + '...');
      
      const response = await axios({
        method: 'POST',
        url: `${this.baseUrl}/${this.projectId}/${this.version}`,
        params: {
          api_key: this.apiKey,
          image: imageUrl,
          format: 'image'
        },
        responseType: 'arraybuffer',
        timeout: 30000 // 30 секунд таймаут
      });

      console.log('✅ Roboflow анализ завершен успешно');
      return {
        success: true,
        data: response.data,
        contentType: response.headers['content-type'] || 'image/jpeg'
      };
    } catch (error) {
      console.error('❌ Ошибка Roboflow API:', error.message);
      
      if (error.response) {
        throw new Error(`Roboflow API Error: ${error.response.status} - ${error.response.statusText}`);
      } else if (error.code === 'ECONNABORTED') {
        throw new Error('Roboflow API timeout');
      } else {
        throw new Error(`Roboflow connection error: ${error.message}`);
      }
    }
  }

  async getAnalysisResults(imageUrl) {
    try {
      const response = await axios({
        method: 'POST',
        url: `${this.baseUrl}/${this.projectId}/${this.version}`,
        params: {
          api_key: this.apiKey,
          image: imageUrl,
          format: 'json'
        },
        timeout: 30000
      });

      return {
        success: true,
        predictions: response.data.predictions || [],
        image: response.data.image || {}
      };
    } catch (error) {
      console.error('❌ Ошибка получения результатов анализа:', error.message);
      throw new Error(`Failed to get analysis results: ${error.message}`);
    }
  }
}

module.exports = new RoboflowService();
