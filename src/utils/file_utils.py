import os
import re
import config as C

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

  return max(model_serials)

def extract_model_serial(model_serial_name):
  """Extract serial from the model name.
  
  Args:
    model_serial_name: Name of the model to extract the serial from.
  
  Returns:
    int: Serial number of the model.
  """
  if(match := re.search(r'*.serial_(*\d).pt', model_serial_name)):
    return int(match.group(1))
  else:
    return None

def list_files_paths(dir):
  """List file paths.

  Args:
    dir (str): Target directory to list.
  
  Returns:
    list: List of file paths inside the directory.
  """
  
  return [os.path.join(dir, f)
          for f in os.listdir(dir) 
          if os.path.isfile(os.path.join(dir, f))
  ]