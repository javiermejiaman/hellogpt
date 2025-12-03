import pytest
import torch
from src.config import Config
from src.environment import get_device
from src.model.transformer import TransformerBlock

@pytest.fixture
def transformer_block():
  return TransformerBlock(Config()).to(get_device()).eval()

@pytest.fixture
def batch():
  return torch.ones(3, dtype=torch.int64).unsqueeze(0)

@pytest.mark.smoke
def test_model_creation(transformer_block):
  assert transformer_block is not None

@pytest.mark.smoke
def test_forward(transformer_block, batch):
  assert transformer_block.forward(batch) is not None