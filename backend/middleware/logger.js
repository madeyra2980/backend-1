// middleware/logger.js
const logger = (req, res, next) => {
  const start = Date.now();
  
  // Логируем входящий запрос
  console.log(`📥 ${req.method} ${req.originalUrl} - ${new Date().toISOString()}`);
  
  // Перехватываем ответ
  const originalSend = res.send;
  res.send = function(data) {
    const duration = Date.now() - start;
    const status = res.statusCode;
    const statusEmoji = status >= 400 ? '❌' : status >= 300 ? '⚠️' : '✅';
    
    console.log(`${statusEmoji} ${req.method} ${req.originalUrl} - ${status} - ${duration}ms`);
    
    return originalSend.call(this, data);
  };
  
  next();
};

module.exports = logger;
