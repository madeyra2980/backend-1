const mongoose = require("mongoose");

const connectDB = async () => {
  try {
    await mongoose.connect("mongodb://127.0.0.1:27017/carchecker", {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    console.log("✅ MongoDB подключена");
  } catch (error) {
    console.error("❌ Ошибка подключения к MongoDB:", error.message);
    process.exit(1);
  }
};

module.exports = connectDB;
