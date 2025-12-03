import torch
import src.config as C

def slide_window(batch: torch.Tensor) -> torch.Tensor: 
  """Truncate tokens to keep max sequence size.

  The tokens are truncated at the beginning to keep the
  most recently generated tokens for context.

  Args:
    batch (Tensor): shape (B, S) - Batch of token sequences.
  
  Returns:
    Tensor: shape (B, S) if sequence length S <= MAX_SEQ_LENGTH,
            else shape (B, MAX_SEQ_LENGTH) for truncated sequences.
  """

  return batch[:, -C.MAX_SEQ_LEN:] if (batch.size(1) > C.MAX_SEQ_LEN)  else batch