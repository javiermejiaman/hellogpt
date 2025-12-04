from unittest.mock import MagicMock, patch
import pytest
from src.config import Config
from src.inference import Inference

@pytest.fixture
def config():
  return Config()

@pytest.fixture
def inference(config):
    with (patch("src.inference.Tokenizer") as mock_tokenizer_class, 
          patch("src.inference.ModelUtils") as mock_model_utils_class):
      mock_tokenizer = MagicMock()
      mock_model_utils = MagicMock()
      mock_model_utils.load_model.return_value = MagicMock()

      mock_tokenizer_class.return_value = mock_tokenizer
      mock_model_utils_class.return_value = mock_model_utils

      yield Inference(config), mock_model_utils, mock_tokenizer

def test_empty_string(inference):
  """Checks generator returns early if prompt is empty."""
  inf, mock_model_utils, mock_tokenizer = inference

  with pytest.raises(StopIteration):
    next(inf.generate(''))

