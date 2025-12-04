from pytest import fixture
import pytest
from transformers import GPT2TokenizerFast
from src.tokenizer.tokenizer import Tokenizer

@fixture
def pretrained_tokenizer():
  return GPT2TokenizerFast.from_pretrained('sshleifer/tiny-gpt2')

@fixture
def tokenizer(pretrained_tokenizer):
  return Tokenizer(pretrained_tokenizer)

@pytest.mark.integration
@pytest.mark.parametrize('prompt', ['test', ''])
def test_encode(prompt, tokenizer):
  """Checks encode works with a string and an empty string."""
  tokenizer.encode([prompt])

@pytest.mark.integration
@pytest.mark.parametrize('token_ids', [[1, 2, 3], []])
def test_decode(token_ids, tokenizer):
  """Checks decode works with a list and an empty list."""
  tokenizer.decode([token_ids])