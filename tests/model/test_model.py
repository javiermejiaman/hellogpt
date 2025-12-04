import pytest
import torch
from src.config import Config
from src.model.model import Model
from src.environment import get_device

@pytest.fixture
def config():
  return Config()

@pytest.fixture
def device():
  return get_device()

@pytest.fixture
def model(device, config):
  return Model(config).to(device).eval()

@pytest.fixture
def batch(device, config):
  return torch.randint(low=0, 
                       high=config.vocab_size, 
                       size=(1, 1)).to(device)

@pytest.mark.smoke
def test_model_creation(model):
  assert model is not None

def test_forward(model, batch):
  assert model.forward(batch) is not None

def test_forward_shape(model, batch, config):
  assert model.forward(batch).shape == (1, 1, config.vocab_size)

def test_device(model, batch, device):
  assert model.forward(batch).device == device