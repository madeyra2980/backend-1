const express = require("express");
const router = express.Router();
const upload = require("../multer/upload");
const { uploadFile, getAllFiles, getFileById, deleteFile, getAnalysisResults, getCarClassification } = require("../controllers/uploadController");
const validation = require("../middleware/validation");

// Загрузка файла с анализом
router.post("/uploads", 
  upload.single("file"), 
  validation.checkFile,
  validation.checkFileType,
  validation.checkFileSize,
  uploadFile
);

// Получение всех файлов с пагинацией
router.get("/files", getAllFiles);

// Получение файла по ID
router.get("/files/:id", getFileById);

// Удаление файла
router.delete("/files/:id", deleteFile);

// Получение результатов анализа в JSON формате
router.get("/files/:id/analysis", getAnalysisResults);

// Получение данных классификации автомобиля
router.get("/files/:id/classification", getCarClassification);

// Health check endpoint
router.get("/health", (req, res) => {
  res.json({
    success: true,
    message: "API is running",
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

module.exports = router;
