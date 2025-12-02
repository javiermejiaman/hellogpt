import torch
from model.model import Model
import src.config as C
from tokenizer.tokenizer import encode, decode
from utils.file_utils import get_model_latest_serial, get_model_path
from utils.sequence_utils import slide_window

model = Model().to(C.DEVICE)
model.load_state_dict(torch.load(get_model_path(get_model_latest_serial()), 
                                 map_location=C.DEVICE))
model.eval()

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
  ).to(C.DEVICE)                                           # (1, S)

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
