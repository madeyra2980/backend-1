const mongoose = require("mongoose");

const fileSchema = new mongoose.Schema({
  filename: { 
    type: String, 
    required: true,
    trim: true
  },
  originalName: {
    type: String,
    required: true,
    trim: true
  },
  url: { 
    type: String, 
    required: true,
    trim: true
  },
  size: {
    type: Number,
    required: true
  },
  mimetype: {
    type: String,
    required: true
  },
  uploadedAt: { 
    type: Date, 
    default: Date.now 
  },
  analysisResults: {
    type: mongoose.Schema.Types.Mixed,
    default: null
  },
  isAnalyzed: {
    type: Boolean,
    default: false
  },
  classification: {
    cleanliness: {
      level: {
        type: String,
        enum: ['clean', 'dirty']
      },
      score: {
        type: Number,
        min: 0,
        max: 1
      },
      brightness: Number,
      contrast: Number,
      colorVariance: Number,
      dirtScore: Number,
      description: String
    },
    integrity: {
      level: {
        type: String,
        enum: ['intact', 'damaged']
      },
      score: {
        type: Number,
        min: 0,
        max: 1
      },
      severityScore: Number,
      defectCount: Number,
      coveragePercentage: Number,
      defectCounts: mongoose.Schema.Types.Mixed,
      description: String
    },
    overallScore: {
      type: Number,
      min: 0,
      max: 1
    },
    recommendation: {
      buy: Boolean,
      confidence: String,
      message: String,
      details: String,
      category: String
    },
    summary: {
      isClean: Boolean,
      isIntact: Boolean,
      totalDefects: Number,
      damageSeverity: Number
    }
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Индексы для быстрого поиска
fileSchema.index({ uploadedAt: -1 });
fileSchema.index({ filename: 1 });
fileSchema.index({ isAnalyzed: 1 });

// Виртуальное поле для времени загрузки в читаемом формате
fileSchema.virtual('uploadedAtFormatted').get(function() {
  return this.uploadedAt.toLocaleString('ru-RU');
});

module.exports = mongoose.model("File", fileSchema);
