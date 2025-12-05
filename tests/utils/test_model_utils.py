import os
from unittest.mock import MagicMock, patch
import pytest
from src.config import Config
from src.model.model import Model
from src.utils.model_utils import ModelUtils
from torch.optim import AdamW
from src.enums import ResourcePath as RP

@pytest.fixture
def config():
  return Config()

@pytest.fixture
def model_utils(config):
  return ModelUtils(config)

@pytest.fixture
def model(config):
  model = Model(config)
  optimizer = AdamW(model.parameters(), lr=0.001)
  return model, optimizer

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

@pytest.fixture
def mock_model():
  with patch('src.utils.model_utils.Model') as mock_class:
    mock_load_state_dict = MagicMock()
    mock_class.load_state_dict = mock_load_state_dict
    yield mock_class, mock_load_state_dict

@pytest.fixture
def mock_torch():
  with patch('src.utils.model_utils.torch') as mock_class:
    mock_save = MagicMock()
    mock_class.save = mock_save
    yield mock_class, mock_save

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

def test_create_new_model_when_checkpoint_not_found(model_utils, mock_model):
  """Checks the model is created fresh if no checkpoints are found."""
  mock_model_class, mock_load_state_dict = mock_model
  with patch.object(model_utils, 'get_model_latest_serial', return_value=None):
    model_utils.load_model()
  mock_load_state_dict.assert_not_called()

def test_save_path_serial_created(mock_os, model_utils, model, mock_torch):
  """Checks if serial is set to 1 if there are no previous checkpoints."""
  model_instance, optimizer = model

  with (patch.object(model_utils, 'get_model_latest_serial', return_value=None),
        patch.object(model_utils, '_get_model_path', return_value='') as mock_model_path):
    model_utils.save_model(model_instance, optimizer, 1, 1.0)
    call_args, call_kwargs = mock_model_path.call_args
    assert call_args[0] == 1

def test_save_path_serial_is_increased(mock_os, model_utils, model, mock_torch):
  """Checks if the serial is increased before saving the model."""
  model_instance, optimizer = model

  with (patch.object(model_utils, 'get_model_latest_serial', return_value=1),
        patch.object(model_utils, '_get_model_path', return_value='') as mock_model_path):
    model_utils.save_model(model_instance, optimizer, 1, 1.0)
    call_args, call_kwargs = mock_model_path.call_args
    assert call_args[0] == 2