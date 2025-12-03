from enum import Enum
import os

class ResourcePath(Enum):
  DATA = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'data'))
  MODEL_STATE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'artifacts', 'checkpoints'))
  TOKENIZER_MODEL = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'artifacts', 'tokenizer'))