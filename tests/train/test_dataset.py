
from unittest.mock import MagicMock, mock_open, patch
import pytest
import torch
from src.config import Config
from src.train.dataset import TextDataset

@pytest.fixture
def config():
  return Config()

@pytest.fixture
def encoded_tokens(config):
  return {
    'half_size': {
      'input_ids': torch.randint(low=0, 
                                 high=config.vocab_size, 
                                 size=(1, int(config.max_seq_len / 2))).tolist()
    },
    'max_size': {
      'input_ids': torch.randint(low=0,
                                 high=config.vocab_size, 
                                 size=(1, config.max_seq_len + 1)).tolist()
    },
    'overflow_1': {
      'input_ids': torch.randint(low=0, 
                                 high=config.vocab_size, 
                                 size=(1, config.max_seq_len + 2)).tolist()
    },
    'overflow_5': {
      'input_ids': torch.randint(low=0, 
                                 high=config.vocab_size, 
                                 size=(1, config.max_seq_len + 6)).tolist()
    }
  }

@pytest.mark.parametrize('seq_size_ds_size', [('half_size', 0), 
                                              ('max_size', 1), 
                                              ('overflow_1', 2)])
@patch('src.train.dataset.Tokenizer')
@patch('src.train.dataset.list_file_paths')
def test_data_splits(mock_list_file_paths, mock_tokenizer_class,
                           encoded_tokens, seq_size_ds_size, config):
  """Checks dataset is split at the correct sequence size. 
  
  Depending on the amount of data, the last sequence may be dropped 
  if its size is shorter than the max_sequence_length. This is intentional: 
  dropping the final short sequence simplifies batching and has
  negligible impact on performance for sufficiently large datasets.
  """
  mock_encode = MagicMock()
  mock_tokenizer_class.return_value = mock_encode

  mock_list_file_paths.return_value = ['test.txt']
  mock_encode.encode.return_value = encoded_tokens[seq_size_ds_size[0]]

  with patch('src.train.dataset.open', mock_open(read_data='test')):
    dataset = TextDataset(config)

    assert len(dataset) == seq_size_ds_size[1]

@patch('src.train.dataset.Tokenizer')
@patch('src.train.dataset.list_file_paths')
def test_sequence_size(mock_list_file_paths, mock_tokenizer_class,
                           encoded_tokens, config):
  """Checks sequence size is fixed at the max_sequence_size."""
  mock_encode = MagicMock()
  mock_tokenizer_class.return_value = mock_encode

  mock_list_file_paths.return_value = ['test.txt']
  mock_encode.encode.return_value = encoded_tokens['overflow_5']

  with patch('src.train.dataset.open', mock_open(read_data='test')):
    dataset = TextDataset(config)

    for sample in dataset:
      seq, target = sample

      assert (seq.size(0) == config.max_seq_len 
              and target.size(0) == config.max_seq_len)