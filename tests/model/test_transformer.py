import pytest

@pytest.fixture
def transformer_block():
  from src.model.transformer import TransformerBlock

  return TransformerBlock()

@pytest.mark.smoke
def test_model_creation(transformer_block):
  assert transformer_block is not None
