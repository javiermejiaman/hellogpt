from unittest.mock import MagicMock, patch
import pytest
import torch
from src.config import Config
from src.environment import get_device, is_tokenizer_model_available, is_model_available

@pytest.fixture
def config():
  return Config()

@pytest.fixture
def device():
  return get_device()

@pytest.fixture
def mock_list_file_paths():
  with patch('src.environment.list_file_paths') as mock_method:
    yield mock_method

@pytest.fixture
def mock_get_model_latest_serial():
  with patch('src.utils.model_utils.ModelUtils') as mock_class:
    mock = MagicMock()
    mock_class.return_value = mock
    yield mock

@pytest.fixture
def tokenizer_model_paths():
  return {
    'vocab_and_merges': [
      '/vocab.json',
      '/merges.txt'
    ],
    'vocab_only': ['/vocab.json'],
    'merges_only': ['/merges.txt']
  }

class TestDevice:

  @pytest.mark.skipif(torch.cuda.is_available(), reason='GPU available')
  def test_cpu_selected_when_only_device_available(self, device):
    """Checks if cpu is used when it's the only device available."""
    assert device == torch.device("cpu")

  @pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU not available')
  def test_cuda_selected_when_available(self, device):
    """Checks if cuda gets selected automatically when available."""
    assert device == torch.device("cuda")

class TestIsTokenizerModelAvailable:

  def test_both_files_available(self, mock_list_file_paths, tokenizer_model_paths):
    """Checks if respond ok when both required files are available."""
    mock_list_file_paths.return_value = tokenizer_model_paths['vocab_and_merges']

    assert is_tokenizer_model_available() is True
  
  @pytest.mark.parametrize('current_path', ['vocab_only', 'merges_only'])
  def test_only_vocab_or_merges_available(self, mock_list_file_paths,
                                          tokenizer_model_paths, current_path):
    """Check if respond properly when only one of the require files is missing."""
    mock_list_file_paths.return_value = tokenizer_model_paths[current_path]

    assert is_tokenizer_model_available() is False

  def test_no_files_available(self, mock_list_file_paths):
    """Checks if respond properly when no files are available."""
    mock_list_file_paths.return_value = []

    assert is_tokenizer_model_available() is False
  
class TestIsModelAvailable:

  def test_model_is_available(self, mock_get_model_latest_serial, config):
    """Checks if respond correctly when model is available."""
    mock_get_model_latest_serial.get_model_latest_serial.return_value = 1

    assert is_model_available(config) is True

  def test_model_is_not_available(self, mock_get_model_latest_serial, config):
    """Checks if respond correctly when model is not available."""
    mock_get_model_latest_serial.get_model_latest_serial.return_value = None

    assert is_model_available(config) is False