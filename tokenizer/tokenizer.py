import config as C
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained(C.TOKENIZER_MODEL_PATH,
                                              bos_token='<bos>',
                                              eos_token='<eos>',
                                              pad_token='<pad>',
                                              unk_token='<unk>')

PAD_TOKEN='<pad>'
PAD_ID=tokenizer.convert_tokens_to_ids('<pad>')

def encode(batch):
  return tokenizer(batch)

def decode(batch):
  return tokenizer.batch_decode(batch)
