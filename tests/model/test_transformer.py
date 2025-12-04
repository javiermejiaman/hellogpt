import pytest
import torch
from src.config import Config
from src.environment import get_device
from src.model.transformer import TransformerBlock

@pytest.fixture
def config():
  return Config()

@pytest.fixture
def device():
  return get_device()

@pytest.fixture
def transformer_block(device, config):
  return TransformerBlock(config).to(device).eval()

@pytest.fixture
def batch(device, config):
  return torch.randn(1, 1, config.d_model).to(device)

@pytest.mark.smoke
def test_model_creation(transformer_block):
  assert transformer_block is not None

def test_forward(transformer_block, batch):
  assert transformer_block.forward(batch) is not None

def test_forward_shape(transformer_block, batch, config):
  assert transformer_block.forward(batch).shape == (1, 1, config.d_model)

def test_device(transformer_block, batch, device):
  assert transformer_block.forward(batch).device == device