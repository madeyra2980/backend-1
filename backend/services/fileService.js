// services/fileService.js
const File = require('../models/File');
const path = require('path');
const fs = require('fs').promises;

class FileService {
  // Сохранение информации о файле в БД
  async saveFileInfo(filename, url, originalName, size, mimetype) {
    try {
      const newFile = new File({
        filename,
        originalName,
        url,
        size,
        mimetype,
        uploadedAt: new Date()
      });
      
      const savedFile = await newFile.save();
      console.log('💾 Файл сохранен в БД:', savedFile._id);
      return savedFile;
    } catch (error) {
      console.error('❌ Ошибка сохранения файла в БД:', error.message);
      throw new Error(`Failed to save file info: ${error.message}`);
    }
  }

  // Получение всех файлов
  async getAllFiles(limit = 10, skip = 0) {
    try {
      const files = await File.find()
        .sort({ uploadedAt: -1 })
        .limit(limit)
        .skip(skip);
      
      const total = await File.countDocuments();
      
      return {
        files,
        total,
        hasMore: skip + files.length < total
      };
    } catch (error) {
      console.error('❌ Ошибка получения файлов:', error.message);
      throw new Error(`Failed to get files: ${error.message}`);
    }
  }

  // Получение файла по ID
  async getFileById(fileId) {
    try {
      const file = await File.findById(fileId);
      if (!file) {
        throw new Error('File not found');
      }
      return file;
    } catch (error) {
      console.error('❌ Ошибка получения файла:', error.message);
      throw new Error(`Failed to get file: ${error.message}`);
    }
  }

  // Удаление файла
  async deleteFile(fileId) {
    try {
      const file = await File.findById(fileId);
      if (!file) {
        throw new Error('File not found');
      }

      // Удаляем физический файл
      const filePath = path.join(__dirname, '../uploads', file.filename);
      try {
        await fs.unlink(filePath);
        console.log('🗑️ Физический файл удален:', filePath);
      } catch (unlinkError) {
        console.warn('⚠️ Не удалось удалить физический файл:', unlinkError.message);
      }

      // Удаляем запись из БД
      await File.findByIdAndDelete(fileId);
      console.log('🗑️ Запись файла удалена из БД:', fileId);
      
      return { success: true };
    } catch (error) {
      console.error('❌ Ошибка удаления файла:', error.message);
      throw new Error(`Failed to delete file: ${error.message}`);
    }
  }

  // Генерация URL файла
  generateFileUrl(req, filename) {
    return `${req.protocol}://${req.get('host')}/uploads/${filename}`;
  }

  // Проверка существования файла
  async fileExists(filename) {
    try {
      const filePath = path.join(__dirname, '../uploads', filename);
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }

  // Обновление данных классификации файла
  async updateFileClassification(fileId, classificationData) {
    try {
      const updatedFile = await File.findByIdAndUpdate(
        fileId,
        {
          classification: classificationData,
          isAnalyzed: true
        },
        { new: true }
      );
      
      if (!updatedFile) {
        throw new Error('File not found');
      }
      
      console.log('📊 Данные классификации обновлены для файла:', fileId);
      return updatedFile;
    } catch (error) {
      console.error('❌ Ошибка обновления классификации:', error.message);
      throw new Error(`Failed to update classification: ${error.message}`);
    }
  }
}

module.exports = new FileService();
