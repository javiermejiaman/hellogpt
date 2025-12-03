import torch
from src.config import Config
from src.tokenizer.tokenizer import Tokenizer
from src.utils.model_utils import load_model
from src.utils.sequence_utils import slide_window
from src.environment import get_device

class Inference:

  def __init__(self, cfg: Config):
    self.cfg = cfg
    self.tokenizer = Tokenizer()
    self._inference_model = load_model()
    self._inference_model.eval()

  def generate(self, prompt):
    """Autoregressive text generation.

    Args:
      prompt (str): Prompt to feed into the model.
    
    Yields:
      str: Generated chunk of text.
    """

    if len(prompt) == 0:
      return None

    batch = self.tokenizer.encode([prompt])['input_ids']     # (1, S)
    batch = torch.tensor(batch, 
                        dtype=torch.long
    ).to(get_device())                                       # (1, S)

    with torch.no_grad():
      batch = slide_window(batch)                            # (1, S)
      
      for _ in range(self.cfg.max_new_tokens):
        logits = self._inference_model(batch)                # (1, S, V)
        next_token_logits = logits[0, -1, :]                 # (V)
        probs = torch.softmax(
          next_token_logits / self.cfg.temperature, dim=-1)  # (V)
        next_token = torch.argmax(probs).view(1, 1)          # (1, 1)
        
        yield self.tokenizer.decode(next_token.tolist())[0]

        batch = torch.cat([batch, next_token], dim=1)        # (1, S+1)
        batch = slide_window(batch)                          # (1, S)
