// middleware/errorHandler.js
const errorHandler = (err, req, res, next) => {
  console.error('❌ Error:', err);

  // Multer ошибки
  if (err.code === 'LIMIT_FILE_SIZE') {
    return res.status(400).json({
      success: false,
      error: 'Файл слишком большой. Максимальный размер: 5MB'
    });
  }

  if (err.code === 'LIMIT_FILE_COUNT') {
    return res.status(400).json({
      success: false,
      error: 'Слишком много файлов. Максимум: 1 файл'
    });
  }

  if (err.code === 'LIMIT_UNEXPECTED_FILE') {
    return res.status(400).json({
      success: false,
      error: 'Неожиданное поле файла'
    });
  }

  // Mongoose ошибки
  if (err.name === 'ValidationError') {
    const errors = Object.values(err.errors).map(e => e.message);
    return res.status(400).json({
      success: false,
      error: 'Ошибка валидации',
      details: errors
    });
  }

  if (err.name === 'CastError') {
    return res.status(400).json({
      success: false,
      error: 'Неверный формат ID'
    });
  }

  // Axios ошибки
  if (err.response) {
    return res.status(err.response.status || 500).json({
      success: false,
      error: 'Ошибка внешнего API',
      details: err.response.data || err.message
    });
  }

  // Общие ошибки
  res.status(err.status || 500).json({
    success: false,
    error: err.message || 'Внутренняя ошибка сервера'
  });
};

module.exports = errorHandler;
