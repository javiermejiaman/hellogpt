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
def transformer_block(request, device, config):
  variant = getattr(request, "param", device)
  return TransformerBlock(config).to(variant).eval()

@pytest.fixture
def batch(request, device, config):
  variant = getattr(request, "param", device)
  return torch.randn(1, 1, config.d_model).to(torch.device(variant))

@pytest.mark.smoke
def test_model_creation(transformer_block):
  assert transformer_block is not None

@pytest.mark.skipif(torch.cuda.is_available(), reason="GPU available")
def test_forward_cpu(transformer_block, batch):
  """Checks forward works on CPU."""
  assert transformer_block.forward(batch) is not None

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
@pytest.mark.parametrize("transformer_block, batch", [
  ("cpu", "cpu"), ("cuda", "cuda")], indirect=True)
def test_forward_cpu_and_gpu(transformer_block, batch):
  """Checks forward works on both CPU and GPU."""
  assert transformer_block.forward(batch) is not None

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
@pytest.mark.parametrize("transformer_block, batch", [
  ("cpu", "cuda"), ("cuda", "cpu")], indirect=True)
def test_forward_mixed_devices(transformer_block, batch):
  """Checks forward fails when mixing CPU and GPU."""
  with pytest.raises(Exception):
    transformer_block.forward(batch)

def test_forward_shape(transformer_block, batch, config):
  """Checks output tensor has expected shape."""
  assert transformer_block.forward(batch).shape == (1, 1, config.d_model)

def test_device(transformer_block, batch, device):
  """Checks output tensor is on the same device."""
  assert transformer_block.forward(batch).device == device