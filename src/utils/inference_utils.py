import torch
from src.config import Config

def slide_window(batch: torch.Tensor, cfg: Config) -> torch.Tensor: 
  """Truncate tokens to keep max sequence size.

  The tokens are truncated at the beginning to keep the
  most recently generated tokens for context.

  Args:
    batch (Tensor): shape (B, S) - Batch of token sequences.
    cfg (Config): Project configuration.
  
  Returns:
    Tensor: shape (B, S) if sequence length S <= max_seq_len,
            else shape (B, max_seq_len) for truncated sequences.
  """

  return batch[:, -cfg.max_seq_len:] if (batch.size(1) > cfg.max_seq_len)  else batch