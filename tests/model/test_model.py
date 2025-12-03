import pytest
import torch
from src.config import Config
from src.model.model import Model
from src.environment import get_device

@pytest.fixture
def config():
  return Config()

@pytest.fixture
def model(config):
  return Model(config).to(get_device()).eval()

@pytest.fixture
def batch(config):
  return torch.randint(low=0, 
                       high=config.vocab_size, 
                       size=(1, 10)).to(get_device())

@pytest.mark.smoke
def test_model_creation(model):
  assert model is not None

@pytest.mark.smoke
def test_forward(model, batch):
  assert model.forward(batch) is not None