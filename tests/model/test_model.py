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
def model(request, device, config):
  variant = getattr(request, "param", device)
  return Model(config).to(variant).eval()

@pytest.fixture
def batch(request, device, config):
  variant = getattr(request, "param", device)
  return torch.randint(low=0, 
                       high=config.vocab_size, 
                       size=(1, 1)).to(variant)

@pytest.mark.smoke
def test_model_creation(model):
  assert model is not None

@pytest.mark.skipif(torch.cuda.is_available(), reason="GPU available")
def test_forward_cpu(model, batch):
  """Checks forward works on CPU."""
  assert model.forward(batch) is not None

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
@pytest.mark.parametrize("model, batch", [
  ("cpu", "cpu"), ("cuda", "cuda")], indirect=True)
def test_forward_cpu_and_gpu(model, batch):
  """Checks forward works on both CPU and GPU."""
  assert model.forward(batch) is not None

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
@pytest.mark.parametrize("model, batch", [
  ("cpu", "cuda"), ("cuda", "cpu")], indirect=True)
def test_forward_mixed_devices(model, batch):
  """Checks forward fails when mixing CPU and GPU."""
  with pytest.raises(Exception):
    model.forward(batch)

def test_forward_shape(model, batch, config):
  """Checks output tensor has expected shape (B, S, V)."""
  assert model.forward(batch).shape == (1, 1, config.vocab_size)

def test_device(model, batch, device):
  """Checks output tensor is on the same device."""
  assert model.forward(batch).device == device