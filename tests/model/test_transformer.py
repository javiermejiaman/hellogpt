import pytest
import torch
from src.config import Config
from src.environment import get_device
from src.model.transformer import TransformerBlock

@pytest.fixture
def config():
  return Config()

@pytest.fixture
def transformer_block(config):
  return TransformerBlock(config).to(get_device()).eval()

@pytest.fixture
def batch():
  return torch.randn(1, 10, 256).to(get_device())

@pytest.mark.smoke
def test_model_creation(transformer_block):
  assert transformer_block is not None

@pytest.mark.smoke
def test_forward(transformer_block, batch):
  assert transformer_block.forward(batch) is not None