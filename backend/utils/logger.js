// utils/logger.js
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

class Logger {
  static info(message, data = null) {
    const timestamp = new Date().toISOString();
    console.log(`${colors.blue}ℹ️  [${timestamp}] INFO: ${message}${colors.reset}`);
    if (data) {
      console.log(`${colors.cyan}   Data:`, data, colors.reset);
    }
  }

  static success(message, data = null) {
    const timestamp = new Date().toISOString();
    console.log(`${colors.green}✅ [${timestamp}] SUCCESS: ${message}${colors.reset}`);
    if (data) {
      console.log(`${colors.cyan}   Data:`, data, colors.reset);
    }
  }

  static warning(message, data = null) {
    const timestamp = new Date().toISOString();
    console.log(`${colors.yellow}⚠️  [${timestamp}] WARNING: ${message}${colors.reset}`);
    if (data) {
      console.log(`${colors.cyan}   Data:`, data, colors.reset);
    }
  }

  static error(message, error = null) {
    const timestamp = new Date().toISOString();
    console.log(`${colors.red}❌ [${timestamp}] ERROR: ${message}${colors.reset}`);
    if (error) {
      console.log(`${colors.red}   Error:`, error, colors.reset);
    }
  }

  static debug(message, data = null) {
    if (process.env.NODE_ENV === 'development') {
      const timestamp = new Date().toISOString();
      console.log(`${colors.magenta}🐛 [${timestamp}] DEBUG: ${message}${colors.reset}`);
      if (data) {
        console.log(`${colors.cyan}   Data:`, data, colors.reset);
      }
    }
  }

  static request(method, url, status, duration) {
    const timestamp = new Date().toISOString();
    const statusColor = status >= 400 ? colors.red : status >= 300 ? colors.yellow : colors.green;
    const emoji = status >= 400 ? '❌' : status >= 300 ? '⚠️' : '✅';
    
    console.log(`${statusColor}${emoji} [${timestamp}] ${method} ${url} - ${status} - ${duration}ms${colors.reset}`);
  }
}

module.exports = Logger;
