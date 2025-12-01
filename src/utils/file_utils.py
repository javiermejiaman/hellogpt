import os
import re
import config as C

def get_model_path(serial):
  return os.path.join(C.CHECKPOINTS_PATH, C.MODEL_NAME, f'serial_{ serial }.pt')

def get_model_latest_serial():
  model_serials = [extract_model_serial(os.path.join(C.CHECKPOINTS_PATH, C.MODEL_NAME, f))
                   for f in os.listdir(C.CHECKPOINTS_PATH, C.MODEL_NAME) 
                   if os.path.isfile(os.path.join(C.CHECKPOINTS_PATH, C.MODEL_NAME, f))
  ]

  return max(model_serials)

def extract_model_serial():
  if(match := re.search(r'*.serial_(*\d).pt', C.MODEL_NAME)):
    return int(match.group(1))
  else:
    return None