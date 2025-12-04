import torch
from torch.utils.data import Dataset
from src.utils.file_utils import list_files_paths
from src.tokenizer.tokenizer import Tokenizer
from src.config import Config
from src.enums import ResourcePath as RP

class TextDataset(Dataset):
  
  def __init__(self, cfg: Config):
    self._cfg = cfg
    self._tokenizer = Tokenizer()

    data = self._load_data()
    tokenized_data = self._encode_data(data)
    self._train_data = self._generate_samples(tokenized_data)
  
  def _load_data(self):
    """Loads data from file system.
    
    Returns:
      str: All data joined together.
    """
    data = []

    for file_path in list_files_paths(RP.DATA.value):
      with open(file_path, 'r', encoding='utf-8') as f:
        data.append(f.read())

    return''.join(data)
  
  def _encode_data(self, data):
    """Tokenizes the data.
    
    Args:
      data (str): String with all training data.
    
    Returns:
      list: Tokenized training data.
    """
    return self._tokenizer.encode([data])["input_ids"][0]
  
  def _generate_samples(self, token_ids):
    """Generates training samples.

    Args:
      token_ids: Training data as a list of tokens.
    
    Returns:
      list: List of training samples.
    """

    data = []
    for i in range(0, len(token_ids) - self._cfg.max_seq_len + 1):
      data.append(token_ids[i:i + self._cfg.max_seq_len])

    return data

  def __len__(self):
    """Returns the total amount of samples.
    
    Returns:
      int: Length of training samples.
    """

    return len(self._train_data)
  
  def __getitem__(self, idx):
    """Gets the training and target sequences.

    The target is the training sequence shifted by one.

    Args:
      idx: Id of the current samples being fetched.
    
    Returns:
      (Tensor, Tensor): Sequence and target.
    """

    seq = torch.tensor(self._train_data[idx][:-1], dtype=torch.long)
    target = torch.tensor(self._train_data[idx][1:], dtype=torch.long)
    return seq, target