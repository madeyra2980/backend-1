// utils/constants.js
const HTTP_STATUS = {
  OK: 200,
  CREATED: 201,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  CONFLICT: 409,
  UNPROCESSABLE_ENTITY: 422,
  INTERNAL_SERVER_ERROR: 500,
  SERVICE_UNAVAILABLE: 503
};

const FILE_CONSTRAINTS = {
  MAX_SIZE: 5 * 1024 * 1024, // 5MB
  ALLOWED_TYPES: ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'],
  ALLOWED_EXTENSIONS: ['.jpg', '.jpeg', '.png', '.webp']
};

const MESSAGES = {
  FILE_UPLOADED: 'Файл успешно загружен',
  FILE_NOT_FOUND: 'Файл не найден',
  FILE_TOO_LARGE: 'Файл слишком большой',
  INVALID_FILE_TYPE: 'Неподдерживаемый тип файла',
  ANALYSIS_SUCCESS: 'Анализ изображения завершен',
  ANALYSIS_FAILED: 'Ошибка анализа изображения',
  SERVER_ERROR: 'Внутренняя ошибка сервера',
  VALIDATION_ERROR: 'Ошибка валидации данных'
};

const API_ENDPOINTS = {
  UPLOAD: '/api/uploads',
  FILES: '/api/files',
  HEALTH: '/api/health'
};

module.exports = {
  HTTP_STATUS,
  FILE_CONSTRAINTS,
  MESSAGES,
  API_ENDPOINTS
};
