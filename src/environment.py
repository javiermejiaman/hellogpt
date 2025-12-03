import torch
from src.utils.file_utils import get_model_latest_serial, list_files_paths
import config as C

def get_device():
  """Gets the device.

  Checks if CUDA is available, fallsback to CPU if not.
  
  Returns:
    device: The current device.
  """

  return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_tokenizer_model_available():
  """Checks if the tokenizer model is available.
  
  Returns:
    bool: True if available, False otherwise.
  """

  file_paths = list_files_paths(C.TOKENIZER_MODEL_PATH)

  exists_vocab = any("vocab.json" in f for f in file_paths)
  exists_merges = any("merges.txt" in f for f in file_paths)

  return exists_vocab and exists_merges

def is_model_available():
  """Checks if a model checkpoint is available.
  
  Returns:
    bool: True if a model serial is available, False otherwise.
  """

  return True if get_model_latest_serial() else False