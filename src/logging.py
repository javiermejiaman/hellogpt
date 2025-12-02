import logging
import config as C

def get_logger(name: str = C.PROJECT_NAME):
  """Provides a logger.

  Args:
    name (str): Logger name.
  
  Returns:
    Logger: logger instance.
  """

  logger = logging.getLogger(name)

  if not logger.handlers:
    logger.setLevel(C.LOGGING_LEVEL)

    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
  
  return logger