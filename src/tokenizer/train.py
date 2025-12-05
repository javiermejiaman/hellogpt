import os
from src.config import Config
from tokenizers import ByteLevelBPETokenizer
from src.utils.file_utils import list_file_paths
from src.logging import get_logger
from src.enums import ResourcePath as RP

def train_tokenizer_model(cfg: Config):
  """Trains the tokenizer model.
  
  The source for this tokenizer model is the training data.
  It's recommeded to retrain the tokenizer model when the 
  data is changed.

  Args:
    cfg (Config): Project configuration.
  """

  log = get_logger(cfg)

  try:
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(files=list_file_paths(RP.DATA.value), 
                    vocab_size=cfg.vocab_size, 
                    min_frequency=cfg.min_freq, 
                    special_tokens=['<pad>', '<unk>', '<bos>', '<eos>']
    )
    
    os.makedirs(RP.TOKENIZER_MODEL.value, exist_ok=True)

    tokenizer.save_model(RP.TOKENIZER_MODEL.value)

    log.debug(f'Tokenizer model saved to "' + RP.TOKENIZER_MODEL.value + '"')
  
  except Exception as e:
    log.error(f'Failed to train tokenizer.', exc_info=True)

    raise e