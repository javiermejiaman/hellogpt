from transformers import GPT2TokenizerFast
from src.enums import ResourcePath as RP

_tokenizer = None

def _get_tokenizer():
  """Gets singleton instances of the tokenizer.

  Returns:
    PreTrainedTokenizerBase: Instances of trained tokenizer.
  """
  
  global _tokenizer

  if _tokenizer is None:
    _tokenizer = GPT2TokenizerFast.from_pretrained(RP.TOKENIZER_MODEL,
                                                  bos_token='<bos>',
                                                  eos_token='<eos>',
                                                  pad_token='<pad>',
                                                  unk_token='<unk>')
  
  return _tokenizer

def encode(batch):
  """Encodes a batch of strings.

  Args:
    batch: Batch of input strings.
  
  Returns:
    Tensor: shape (B, S) - Batch of token sequences.
  """

  tokenizer = _get_tokenizer()

  return tokenizer(batch)

def decode(batch):
  """Decodes a batch of token sequences.

  Args:
    batch (Tensor): shape (B, S) - Batch of token sequences.
  
  Results:
    list: List of decoded strings.
  """

  tokenizer = _get_tokenizer()

  return tokenizer.batch_decode(batch)
