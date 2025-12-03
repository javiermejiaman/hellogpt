from transformers import GPT2TokenizerFast
from src.enums import ResourcePath as RP

class Tokenizer:

  def __init__(self):
    self._tokenizer = GPT2TokenizerFast.from_pretrained(
      RP.TOKENIZER_MODEL,
      bos_token='<bos>',
      eos_token='<eos>',
      pad_token='<pad>',
      unk_token='<unk>')

def encode(self, batch):
  """Encodes a batch of strings.

  Args:
    batch: Batch of input strings.
  
  Returns:
    Tensor: shape (B, S) - Batch of token sequences.
  """

  tokenizer = self._tokenizer()

  return tokenizer(batch)

def decode(self, batch):
  """Decodes a batch of token sequences.

  Args:
    batch (Tensor): shape (B, S) - Batch of token sequences.
  
  Results:
    list: List of decoded strings.
  """

  tokenizer = self._tokenizer()

  return tokenizer.batch_decode(batch)
