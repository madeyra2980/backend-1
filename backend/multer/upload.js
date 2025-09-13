const multer = require("multer");
const path = require("path");
const { FILE_CONSTRAINTS } = require("../utils/constants");

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/");
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(
      null,
      file.fieldname + "-" + uniqueSuffix + path.extname(file.originalname)
    );
  },
});

const fileFilter = (req, file, cb) => {
  const allowedTypes = /jpeg|jpg|png|webp/;
  const ext = allowedTypes.test(path.extname(file.originalname).toLowerCase());
  const mime = allowedTypes.test(file.mimetype);

  if (ext && mime) {
    cb(null, true);
  } else {
    cb(new Error("Только изображения (jpeg, jpg, png, webp)"));
  }
};

const upload = multer({ 
  storage, 
  fileFilter,
  limits: {
    fileSize: parseInt(process.env.MAX_FILE_SIZE) || FILE_CONSTRAINTS.MAX_SIZE,
    files: 1 // Только один файл за раз
  }
});

module.exports = upload;
