from loguru import logger
import sys
import os

LOG_FILE = os.path.join("logs", "dashboard.log")
os.makedirs("logs", exist_ok=True)

# Configure loguru
logger.remove()  # Remove default stderr logger
logger.add(sys.stderr, level="INFO")  # Console logging
logger.add(LOG_FILE, rotation="1 MB", retention="7 days", level="DEBUG")  # File logging

# Export the logger
log = logger