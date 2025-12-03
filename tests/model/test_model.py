
import pytest

@pytest.fixture
def model():
  from src.model.model import Model

  return Model()

@pytest.mark.smoke
def test_model_creation(model):
  assert model is not None
