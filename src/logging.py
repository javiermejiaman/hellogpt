import logging
from src.config import Config

def get_logger(cfg: Config):
  """Provides a logger.

  Args:
    cfg (Config): Project configuration.
  
  Returns:
    Logger: logger instance.
  """

  logger = logging.getLogger(cfg.project_name)

  if not logger.handlers:
    logger.setLevel(cfg.logging_level)

    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
  
  return logger