import pytest
import torch
from src.config import Config
from src.model.model import Model
from src.environment import get_device

@pytest.fixture
def model():
  return Model(Config()).to(get_device()).eval()

@pytest.fixture
def batch():
  return torch.ones(3, dtype=torch.int64).unsqueeze(0)

@pytest.mark.smoke
def test_model_creation(model):
  assert model is not None

@pytest.mark.smoke
def test_forward(model, batch):
  assert model.forward(batch) is not None