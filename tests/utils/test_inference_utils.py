import pytest
import torch
from src.config import Config
from src.environment import get_device
from src.utils.inference_utils import slide_window

@pytest.fixture
def config():
  return Config()

@pytest.fixture
def device():
  return get_device()

@pytest.fixture
def batches(device, config):
  return {
    'half_size': torch.randint(low=0,
                              high=config.vocab_size,
                              size=(1, config.max_seq_len)).to(device),
    'max_size': torch.randint(low=0,
                              high=config.vocab_size,
                              size=(1, config.max_seq_len)).to(device),
    'overflow': torch.randint(low=0,
                              high=config.vocab_size,
                              size=(1, config.max_seq_len + 1)).to(device) 
  }

@pytest.mark.parametrize('sequence_size', ['half_size', 'max_size'])
def test_no_slide(batches, sequence_size, config):
  """Checks no changes to sequence if within max sequence size"""
  batch = batches[sequence_size]
  end_index = batch.size(1) - 1
  assert slide_window(batch, config)[0][0] == batch[0][0]
  assert slide_window(batch, config)[0][end_index] == batch[0][end_index]
  assert slide_window(batch, config)[0].size(0) == batch[0].size(0)

def test_slide_window_applied(batches, config):
  """Checks truncation if max sequence size exceeded"""
  batch = batches['overflow']
  new_start_index = batch.size(1) - config.max_seq_len
  last_index = batch.size(1) - 1
  assert slide_window(batch, config)[0][0] == batch[0][new_start_index]
  assert slide_window(batch, config)[0][config.max_seq_len - 1] == batch[0][last_index]
  assert slide_window(batch, config)[0].size(0) == config.max_seq_len