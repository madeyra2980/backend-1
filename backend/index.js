const express = require("express");
const path = require("path");
const cors = require("cors");
require("dotenv").config();

const connectDB = require("./config/db");
const uploadRoutes = require("./routes/uploadRoutes");
const errorHandler = require("./middleware/errorHandler");
const logger = require("./middleware/logger");
const Logger = require("./utils/logger");

const app = express();
const PORT = process.env.PORT || 1015;

// Подключаем MongoDB
connectDB();

// CORS настройки
app.use(cors({
  origin: process.env.CORS_ORIGIN || "http://localhost:3000",
  credentials: true
}));

// Middleware (обязательно ДО роутов)
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Логирование запросов
app.use(logger);

// Статическая папка для загруженных файлов
app.use("/uploads", express.static(path.join(__dirname, "uploads")));

// Главная страница
app.get("/", (req, res) => {
  res.json({
    success: true,
    message: "Car Checker API",
    version: "1.0.0",
    endpoints: {
      upload: "POST /api/uploads",
      files: "GET /api/files",
      fileById: "GET /api/files/:id",
      deleteFile: "DELETE /api/files/:id",
      analysis: "GET /api/files/:id/analysis",
      health: "GET /api/health"
    }
  });
});

// API роуты
app.use("/api", uploadRoutes);

// 404 обработчик
app.use((req, res) => {
  res.status(404).json({
    success: false,
    message: "Endpoint not found",
    path: req.originalUrl,
    availableEndpoints: [
      "POST /api/uploads",
      "GET /api/files",
      "GET /api/files/:id",
      "DELETE /api/files/:id",
      "GET /api/files/:id/analysis",
      "GET /api/health"
    ]
  });
});

// Обработчик ошибок (должен быть последним)
app.use(errorHandler);

// Graceful shutdown
process.on('SIGTERM', () => {
  Logger.info('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  Logger.info('SIGINT received, shutting down gracefully');
  process.exit(0);
});

app.listen(PORT, () => {
  Logger.success(`🚀 Сервер запущен: http://localhost:${PORT}`);
  Logger.info(`📁 Статические файлы: http://localhost:${PORT}/uploads`);
  Logger.info(`🔍 API документация: http://localhost:${PORT}/`);
});
