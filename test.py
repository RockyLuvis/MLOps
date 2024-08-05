import sys
from src.logger.logging import logging

try:
    logging.info("This is my testing")
    #a=1/0
except Exception as e:

    print(sys.exc_info())