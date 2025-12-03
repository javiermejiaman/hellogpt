from dataclasses import dataclass
from pydantic import BaseModel
import logging

@dataclass
class Config(BaseModel):

  # Project
  project_name: str = 'HelloGPT'
  show_banner: bool = True

  # Logging
  logging_level: int = logging.INFO

  # Tokenization
  vocab_size: int = 8000
  min_freq: int = 2

  # Model
  model_name: str = 'hellogpt'
  max_seq_len: int = 128
  d_model: int = 256
  d_ffn: int = 1024
  num_layers: int = 4
  num_heads: int = 4

  # Inference
  max_new_tokens: int = 100
  temperature: float = 0.8

  # Training
  learning_rate: float = 1e-4
  batch_size: int = 32
  epochs: int = 50
  dropout: float = 0.1
  grad_clip: float = 1.0
  train_to_valid_ratio: float = 0.9
