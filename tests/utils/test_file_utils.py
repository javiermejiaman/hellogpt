import os
from unittest.mock import MagicMock, patch
import pytest

from src.utils.file_utils import list_file_paths

@pytest.fixture
def mock_os():
  with patch('src.utils.file_utils.os') as mock_class:
    mock_path = MagicMock()
    mock_class.path = mock_path
    yield mock_class, mock_path

def test_return_empty_if_not_found(mock_os):
  """Checks empty list is returned if directory is not found."""
  mock_os, mock_path = mock_os

  mock_path.isdir.return_value = False

  assert list_file_paths('test') == []

def test_return_empty_if_not_files_found(mock_os):
  """Checks empty list is returned if directory doesn't have files."""
  mock_os, mock_path = mock_os

  mock_path.isdir.return_value = True
  mock_os.listdir.return_value = []

  assert list_file_paths('test') == []

@pytest.mark.integration
def test_return_file_paths_if_not_empty():
  """Checks lists the file paths if directory have files."""

  assert len(list_file_paths(os.path.join(__file__, '..','..'))) > 0