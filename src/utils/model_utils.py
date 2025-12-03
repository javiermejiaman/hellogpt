import os
import re
import torch
import src.config as C
from src.model.model import Model
from src.utils.file_utils import list_files_paths
from src.logging import get_logger

log = get_logger()

def get_model_path(serial):
  """Gets the model path.

  Args:
    serial (int): Serial number of the target model.

  Returns:
    str: Path to the model.
  """

  return os.path.join(C.CHECKPOINTS_PATH, 
                      C.MODEL_NAME, 
                      f'serial_{ serial }.pt')

def get_model_latest_serial():
  """Gets the latest serial of the model.

  Returns:
    int: Latest serial of the model.
  """

  model_checkpoints_path = os.path.join(C.CHECKPOINTS_PATH, 
                                        C.MODEL_NAME)
  
  model_serials = [extract_model_serial(f) 
                   for f in list_files_paths(model_checkpoints_path)
  ]

  if (len(model_serials) == 0 or all(None == s for s in model_serials)):
    return None

  return max(s for s in model_serials if s is not None)

def extract_model_serial(model_serial_name):
  """Extract serial from the model name.
  
  Args:
    model_serial_name: Name of the model to extract the serial from.
  
  Returns:
    int: Serial number of the model.
  """
  if(match := re.search(r'.*serial_(\d*).pt', model_serial_name)):
    return int(match.group(1))
  else:
    return None

def load_model():
  """Loads model.
  
  Returns:
    Model: Model instance.
  """

  from src.environment import get_device
  
  model = Model()
  
  if serial := get_model_latest_serial():
    checkpoint = torch.load(get_model_path(serial), map_location="cpu")
    model.load_state_dict(checkpoint['model_state'])

  return model.to(get_device())

def save_model(model, optimizer, epoch, train_loss):
  """Saves model.

  Saves the model state, optimizer state, epoch and training 
  loss of the latest epoch.
  
  Args:
    model (Model): Model instance.
    optimizer (AdamW): Model optimizer instance.
    epoch (int): Current training epoch.
    train_loss (float): Training loss of the last epoch.
  """

  try:
    os.makedirs(os.path.join(C.CHECKPOINTS_PATH, C.MODEL_NAME), 
                exist_ok=True)
    
    checkpoint = {
      "model_state": model.state_dict(),
      "optimizer_state": optimizer.state_dict(),
      "epoch": epoch,
      "loss": train_loss
    }
    
    latest_serial = get_model_latest_serial()
    next_serial = latest_serial + 1 if latest_serial else 1
    model_path = get_model_path(next_serial)
    
    torch.save(checkpoint, model_path)

    log.debug(f'Model state saved to "' + model_path + '"')
  
  except Exception as e:
    log.error(f'Failed to save model "' + C.MODEL_NAME 
              + '" serial #' + next_serial + '.', exc_info=True)
    
    raise e