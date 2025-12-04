import torch
from src.config import Config
from src.tokenizer.tokenizer import Tokenizer
from src.utils.model_utils import ModelUtils
from src.utils.inference_utils import slide_window
from src.environment import get_device

class Inference:

  def __init__(self, cfg: Config):
    self._cfg = cfg
    
    model_utils = ModelUtils(cfg)
    self._inference_model = model_utils.load_model()
    self._inference_model.eval()

    self._tokenizer = Tokenizer()

  def generate(self, prompt):
    """Autoregressive text generation.

    Args:
      prompt (str): Prompt to feed into the model.
    
    Yields:
      str: Generated chunk of text.
    """

    if len(prompt) == 0:
      return None

    batch = self._tokenizer.encode([prompt])['input_ids']       # (1, S)
    batch = torch.tensor(batch, 
                        dtype=torch.int64
    ).to(get_device())                                          # (1, S)

    with torch.no_grad():
      batch = slide_window(batch, self._cfg)                     # (1, S)
      
      for _ in range(self._cfg.max_new_tokens):
        logits = self._inference_model(batch)                   # (1, S, V)
        next_token_logits = logits[0, -1, :]                    # (V)
        probs = torch.softmax(
          next_token_logits / self._cfg.temperature, dim=-1)    # (V)
        next_token = torch.argmax(probs).view(1, 1)             # (1, 1)
        
        yield self._tokenizer.decode(next_token.tolist())[0]

        batch = torch.cat([batch, next_token], dim=1)           # (1, S+1)
        batch = slide_window(batch, self._cfg)                   # (1, S)
