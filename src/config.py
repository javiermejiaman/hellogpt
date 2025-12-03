import logging
import os

# Project
PROJECT_NAME: str = 'HelloGPT'
SHOW_BANNER: bool = True

# Logging
LOGGING_LEVEL: any = logging.INFO

# Tokenization
VOCAB_SIZE: int = 8000
MIN_FREQ: int = 2

# Model
MODEL_NAME: str = 'hellogpt'
MAX_SEQ_LEN: int = 128
D_MODEL: int = 256
D_FF: int = D_MODEL * 4
NUM_LAYERS: int = 4
NUM_HEADS: int = 4

# Inference
MAX_NEW_TOKENS: int = 100
TEMPERATURE: float = 0.8

# Training
LEARNING_RATE: float = 1e-4
BATCH_SIZE: int = 32
EPOCHS: int = 50
DROPOUT: float = 0.1
GRAD_CLIP: float = 1.0
TRAIN_TO_VALID_RATIO: float = 0.9

# Static  
DATA_PATH: str = os.path.abspath(os.path.join(
  os.path.dirname(__file__), '..', 'data'))
CHECKPOINTS_PATH: str = os.path.abspath(
  os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'checkpoints'))
TOKENIZER_MODEL_PATH: str = os.path.abspath(
  os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'tokenizer'))
