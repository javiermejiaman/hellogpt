import src.config as C
from transformers import GPT2TokenizerFast

_tokenizer = None

def get_tokenizer():
  """Gets singleton instances of the tokenizer.

  Returns:
    PreTrainedTokenizerBase: Instances of trained tokenizer.
  """
  
  global _tokenizer

  if _tokenizer is None:
    _tokenizer = GPT2TokenizerFast.from_pretrained(C.TOKENIZER_MODEL_PATH,
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

  tokenizer = get_tokenizer()

  return tokenizer(batch)

def decode(batch):
  """Decodes a batch of token sequences.

  Args:
    batch (Tensor): shape (B, S) - Batch of token sequences.
  
  Results:
    list: List of decoded strings.
  """

  tokenizer = get_tokenizer()

  return tokenizer.batch_decode(batch)
