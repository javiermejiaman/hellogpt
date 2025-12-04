from unittest.mock import MagicMock, patch
from src.config import Config
from src.tokenizer.train import train_tokenizer_model
import pytest

@pytest.fixture
def config():
  return Config()

@pytest.fixture
def mock_bpe_tokenizer():
  with patch("src.tokenizer.train.ByteLevelBPETokenizer") as mock_class:
    mock = MagicMock()
    mock_class.return_value = mock
    yield mock

@pytest.fixture
def mock_makedirs():
  with patch("src.tokenizer.train.os.makedirs") as mock_func:
    mock = MagicMock()
    mock_func.return_value = mock
    yield mock_func

def test_train_tokenizer_model(mock_bpe_tokenizer, mock_makedirs, config):
  """Checks if expected functions are called in the training process."""
  train_tokenizer_model(config)

  mock_makedirs.assert_called_once()

  mock_bpe_tokenizer.train.assert_called_once()
  mock_bpe_tokenizer.save_model.assert_called_once()

def test_exception(mock_bpe_tokenizer, mock_makedirs, config):
  """Checks exception is raised from the tokenizer."""
  mock_makedirs.side_effect = Exception()

  with pytest.raises(Exception):
    train_tokenizer_model(config)