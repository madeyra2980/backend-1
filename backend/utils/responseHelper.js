// utils/responseHelper.js
class ResponseHelper {
  // Успешный ответ
  static success(res, data = null, message = 'Success', statusCode = 200) {
    return res.status(statusCode).json({
      success: true,
      message,
      data,
      timestamp: new Date().toISOString()
    });
    
  }

  // Ошибка
  static error(res, message = 'Error', statusCode = 500, details = null) {
    return res.status(statusCode).json({
      success: false,
      message,
      details,
      timestamp: new Date().toISOString()
    });
  }

  // Ответ с изображением
  static image(res, imageData, contentType = 'image/jpeg', filename = 'result.jpg') {
    res.set({
      'Content-Type': contentType,
      'Content-Disposition': `inline; filename="${filename}"`,
      'Cache-Control': 'public, max-age=3600'
    });
    return res.send(imageData);
  }

  // Ответ с JSON данными
  static json(res, data, statusCode = 200) {
    return res.status(statusCode).json({
      success: true,
      data,
      timestamp: new Date().toISOString()
    });
  }

  // Пагинация
  static paginated(res, data, page = 1, limit = 10, total = 0) {
    const totalPages = Math.ceil(total / limit);
    
    return res.json({
      success: true,
      data,
      pagination: {
        currentPage: page,
        totalPages,
        totalItems: total,
        itemsPerPage: limit,
        hasNext: page < totalPages,
        hasPrev: page > 1
      },
      timestamp: new Date().toISOString()
    });
  }
}

module.exports = ResponseHelper;
