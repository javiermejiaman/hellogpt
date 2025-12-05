from unittest.mock import MagicMock, patch
import pytest
from src.config import Config
from src.utils.model_utils import ModelUtils

@pytest.fixture
def config():
  return Config()

@pytest.fixture
def model_utils(config):
  return ModelUtils(config)

@pytest.fixture
def mock_os():
  with patch('src.utils.model_utils.os') as mock_class:
    mock_path = MagicMock()
    mock_class.path = mock_path
    yield mock_class, mock_path

@pytest.fixture
def mock_list_file_paths():
  with patch('src.utils.model_utils.list_file_paths') as mock_method:
    yield mock_method

def test_none_if_model_have_no_checkpoints(model_utils, mock_os, 
                                          mock_list_file_paths):
  """Checks None is returned if the model have no checkpoits."""
  mock_list_file_paths.return_value = []

  assert model_utils.get_model_latest_serial() is None

def test_model_serial_is_extracted(model_utils, mock_os, 
                                          mock_list_file_paths):
  """Checks serial is extracter following the pattern serial_{n}.pt"""
  mock_list_file_paths.return_value = ['serial_1.pt']

  assert model_utils.get_model_latest_serial() == 1

def test_model_latest_serial_is_selected(model_utils, mock_os, 
                                          mock_list_file_paths):
  """Checks order of files do not affect results."""
  mock_list_file_paths.return_value = ['serial_2.pt','serial_1.pt']

  assert model_utils.get_model_latest_serial() == 2