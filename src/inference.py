import torch
import src.config as C
from src.tokenizer.tokenizer import encode, decode
from src.utils.file_utils import load_model
from src.utils.sequence_utils import slide_window
from src.environment import get_device

model = load_model()

def generate(prompt):
  """Autoregressive text generation.

  Args:
    prompt (str): Prompt to feed into the model.
  
  Yields:
    str: Generated chunk of text.
  """

  batch = encode([prompt])['input_ids']                    # (1, S)
  batch = torch.tensor(batch, 
                       dtype=torch.long
  ).to(get_device())                                       # (1, S)

  with torch.no_grad():
    batch = slide_window(batch)                            # (1, S)
    
    for _ in range(C.MAX_NEW_TOKENS):
      logits = model(batch)                                # (1, S, V)
      next_token_logits = logits[0, -1, :]                 # (V)
      probs = torch.softmax(next_token_logits 
                            / C.TEMPERATURE, dim=-1)       # (V)
      next_token = torch.argmax(probs).view(1, 1)          # (1, 1)
      
      yield decode(next_token.tolist())[0]

      batch = torch.cat([batch, next_token], dim=1)        # (1, S+1)
      batch = slide_window(batch)                          # (1, S)
