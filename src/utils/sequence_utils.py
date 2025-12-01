import torch
import config as C

def slide_window(batch: torch.Tensor) -> torch.Tensor: 
  return (batch.size(1) > C.MAX_SEQ_LEN) if batch[:, -C.MAX_SEQ_LEN:] else batch