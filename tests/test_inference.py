from unittest.mock import MagicMock, patch
import pytest
import torch
from src.config import Config
from src.environment import get_device
from src.inference import Inference

@pytest.fixture
def config():
  return Config(max_new_tokens=1)

@pytest.fixture
def device():
  return get_device()

@pytest.fixture
def encoded_tokens(device, config):
  return {
    'half_size': {
      'input_ids': torch.randint(low=0, 
                                 high=config.vocab_size, 
                                 size=(1, int(config.max_seq_len / 2))).to(device)
    },
    'max_size': {
      'input_ids': torch.randint(low=0,
                                 high=config.vocab_size, 
                                 size=(1, config.max_seq_len)).to(device)
    },
    'overflow': {
      'input_ids': torch.randint(low=0, 
                                 high=config.vocab_size, 
                                 size=(1, config.max_seq_len + 1)).to(device)
    }
  }

@pytest.fixture
def model_output(device, config):
  return torch.randint(low=0, 
                       high=config.vocab_size, 
                       size=(1, 1, config.vocab_size)).to(device)

@pytest.fixture
def decoded_tokens():
  return ["te", "st"]


@pytest.fixture
def inference(config):
    with (patch('src.inference.Tokenizer') as mock_tokenizer_class, 
          patch('src.inference.ModelUtils') as mock_model_utils_class):
      mock_tokenizer = MagicMock()
      mock_tokenizer_class.return_value = mock_tokenizer

      mock_inference_model = MagicMock()
      
      mock_model_utils = MagicMock()
      mock_model_utils.load_model.return_value = mock_inference_model
      mock_model_utils_class.return_value = mock_model_utils

      yield Inference(config), mock_inference_model, mock_tokenizer

def test_empty_string(inference):
  """Checks generator returns early if prompt is empty."""
  inf, mock_inference_model, mock_tokenizer = inference

  with pytest.raises(StopIteration):
    next(inf.generate(''))

@pytest.mark.parametrize('sequence_size', ['half_size', 'max_size'])
def test_encoded_tokens_size_ok(inference, encoded_tokens, 
                                   decoded_tokens, model_output,
                                   sequence_size):
  """Checks if inference works well with expected sequence size."""
  inf, mock_inference_model, mock_tokenizer = inference

  mock_tokenizer.encode.return_value = {
    'input_ids': encoded_tokens[sequence_size]['input_ids'].tolist()
  }
  mock_tokenizer.decode.return_value = decoded_tokens
  mock_inference_model.return_value = model_output

  next(inf.generate('test'))

  call_args, call_kwargs = mock_inference_model.call_args

  assert torch.equal(call_args[0], encoded_tokens[sequence_size]['input_ids'])

def test_encoded_tokens_size_overflow(inference, encoded_tokens, 
                                   decoded_tokens, model_output):
  """Checks if inference truncates inputs that exceed max seq size."""
  inf, mock_inference_model, mock_tokenizer = inference

  mock_tokenizer.encode.return_value = {
    'input_ids': encoded_tokens['overflow']['input_ids'].tolist()
  }
  mock_tokenizer.decode.return_value = decoded_tokens
  mock_inference_model.return_value = model_output

  next(inf.generate('test'))

  call_args, call_kwargs = mock_inference_model.call_args

  assert call_args[0].size(1) < encoded_tokens['overflow']['input_ids'].size(1)
