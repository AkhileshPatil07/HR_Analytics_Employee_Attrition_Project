"""
Centralized logging configuration.
Every component imports `logging` after this module runs, so every
log line across the pipeline goes to the same timestamped file inside
logs/ as well as the console — makes debugging a failed run much
easier than scattered print() statements.
"""

import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)
