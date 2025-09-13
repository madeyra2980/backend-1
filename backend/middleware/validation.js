// middleware/validation.js
const validation = {
  // Проверка наличия файла
  checkFile: (req, res, next) => {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'Файл не загружен'
      });
    }
    next();
  },

  // Проверка типа файла
  checkFileType: (req, res, next) => {
    if (req.file) {
      const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
      if (!allowedTypes.includes(req.file.mimetype)) {
        return res.status(400).json({
          success: false,
          error: 'Неподдерживаемый тип файла. Разрешены: JPEG, JPG, PNG, WEBP'
        });
      }
    }
    next();
  },

  // Проверка размера файла
  checkFileSize: (req, res, next) => {
    if (req.file && req.file.size > 5 * 1024 * 1024) { // 5MB
      return res.status(400).json({
        success: false,
        error: 'Файл слишком большой. Максимальный размер: 5MB'
      });
    }
    next();
  }
};

module.exports = validation;
