import pytest
from src.config import Config
from src.logging import get_logger

@pytest.fixture
def config():
  return Config()

@pytest.mark.smoke
def test_creation(config):
  get_logger(config)