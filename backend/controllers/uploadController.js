// controllers/uploadController.js
const path = require('path');
const fileService = require('../services/fileService');
const roboflowService = require('../services/roboflowService');
const ResponseHelper = require('../utils/responseHelper');
const Logger = require('../utils/logger');
const { MESSAGES } = require('../utils/constants');

const uploadFile = async (req, res) => {
  try {
    Logger.info('Начинаем загрузку файла', { filename: req.file?.filename });

    // Генерируем URL файла
    const fileUrl = fileService.generateFileUrl(req, req.file.filename);
    Logger.info('URL файла сгенерирован', { fileUrl });

    // Сохраняем информацию о файле в БД
    const savedFile = await fileService.saveFileInfo(
      req.file.filename, 
      fileUrl, 
      req.file.originalname, 
      req.file.size, 
      req.file.mimetype
    );
    Logger.success('Файл сохранен в БД', { fileId: savedFile._id });

    // Анализируем изображение через Roboflow
    try {
      const analysisResult = await roboflowService.analyzeImage(fileUrl);
      Logger.success('Анализ изображения завершен');

      // Сохраняем данные классификации в БД, если они есть
      if (analysisResult.classification) {
        await fileService.updateFileClassification(savedFile._id, analysisResult.classification);
        Logger.success('Данные классификации сохранены в БД');
      }

      // Возвращаем изображение с результатами анализа (с разметкой ржавчины и царапин)
      return ResponseHelper.image(
        res, 
        analysisResult.data, 
        analysisResult.contentType, 
        `analyzed_${req.file.filename}`
      );
    } catch (analysisError) {
      Logger.warning('Анализ недоступен, возвращаем оригинальное изображение', analysisError.message);
      
      // Если анализ не работает, возвращаем оригинальное изображение
      const fs = require('fs');
      const imagePath = path.join(__dirname, '../uploads', req.file.filename);
      const imageData = fs.readFileSync(imagePath);
      
      return ResponseHelper.image(
        res, 
        imageData, 
        req.file.mimetype, 
        req.file.filename
      );
    }

  } catch (error) {
    Logger.error('Ошибка при загрузке/анализе файла', error);
    return ResponseHelper.error(res, MESSAGES.ANALYSIS_FAILED, 500, error.message);
  }
};

// Получение всех файлов
const getAllFiles = async (req, res) => {
  try {
    const { page = 1, limit = 10 } = req.query;
    const skip = (page - 1) * limit;

    const result = await fileService.getAllFiles(parseInt(limit), parseInt(skip));
    Logger.info('Получен список файлов', { count: result.files.length });

    return ResponseHelper.paginated(res, result.files, parseInt(page), parseInt(limit), result.total);

  } catch (error) {
    Logger.error('Ошибка получения файлов', error);
    return ResponseHelper.error(res, 'Ошибка получения файлов', 500, error.message);
  }
};

// Получение файла по ID
const getFileById = async (req, res) => {
  try {
    const { id } = req.params;
    const file = await fileService.getFileById(id);
    
    Logger.info('Файл найден', { fileId: id });
    return ResponseHelper.success(res, file, 'Файл найден');

  } catch (error) {
    Logger.error('Ошибка получения файла', error);
    return ResponseHelper.error(res, MESSAGES.FILE_NOT_FOUND, 404, error.message);
  }
};

// Удаление файла
const deleteFile = async (req, res) => {
  try {
    const { id } = req.params;
    await fileService.deleteFile(id);
    
    Logger.success('Файл удален', { fileId: id });
    return ResponseHelper.success(res, null, 'Файл успешно удален');

  } catch (error) {
    Logger.error('Ошибка удаления файла', error);
    return ResponseHelper.error(res, 'Ошибка удаления файла', 500, error.message);
  }
};

// Получение результатов анализа в JSON формате
const getAnalysisResults = async (req, res) => {
  try {
    const { id } = req.params;
    const file = await fileService.getFileById(id);
    
    const results = await roboflowService.getAnalysisResults(file.url);
    Logger.info('Результаты анализа получены', { fileId: id });

    return ResponseHelper.success(res, results, 'Результаты анализа получены');

  } catch (error) {
    Logger.error('Ошибка получения результатов анализа', error);
    return ResponseHelper.error(res, 'Ошибка получения результатов анализа', 500, error.message);
  }
};

// Получение данных классификации автомобиля
const getCarClassification = async (req, res) => {
  try {
    const { id } = req.params;
    const file = await fileService.getFileById(id);
    
    if (!file.classification) {
      return ResponseHelper.error(res, 'Данные классификации не найдены', 404, 'Файл не был проанализирован');
    }

    Logger.info('Данные классификации получены', { fileId: id });
    return ResponseHelper.success(res, file.classification, 'Данные классификации получены');

  } catch (error) {
    Logger.error('Ошибка получения данных классификации', error);
    return ResponseHelper.error(res, 'Ошибка получения данных классификации', 500, error.message);
  }
};

module.exports = { 
  uploadFile, 
  getAllFiles, 
  getFileById, 
  deleteFile, 
  getAnalysisResults,
  getCarClassification
};
