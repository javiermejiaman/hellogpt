import torch
import os

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

# Static  
DATA_PATH: str = os.path.abspath(os.path.join(
  os.path.dirname(__file__), '..', 'data'))
CHECKPOINTS_PATH: str = os.path.abspath(
  os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'checkpoints'))
TOKENIZER_MODEL_PATH: str = os.path.abspath(
  os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'tokenizer'))

# Hardware
DEVICE: any = torch.device("cuda" if torch.cuda.is_available() else "cpu")
