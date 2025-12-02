import src.config as C
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained(C.TOKENIZER_MODEL_PATH,
                                              bos_token='<bos>',
                                              eos_token='<eos>',
                                              pad_token='<pad>',
                                              unk_token='<unk>')

PAD_TOKEN='<pad>'
PAD_ID=tokenizer.convert_tokens_to_ids('<pad>')

def encode(batch):
  """Encodes a batch of strings.

  Args:
    batch: Batch of input strings.
  
  Returns:
    Tensor: shape (B, S) - Batch of token sequences.
  """

  return tokenizer(batch)

def decode(batch):
  """Decodes a batch of token sequences.

  Args:
    batch (Tensor): shape (B, S) - Batch of token sequences.
  
  Results:
    list: List of decoded strings.
  """

  return tokenizer.batch_decode(batch)
