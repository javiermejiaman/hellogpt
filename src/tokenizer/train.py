import os
from src.config import Config
from tokenizers import ByteLevelBPETokenizer
from src.utils.file_utils import list_files_paths
from src.logging import get_logger
from src.enums import ResourcePath as RP

def train_tokenizer_model(cfg: Config):
  """Trains the tokenizer model.
  
  The source for this tokenizer model is the training data.
  It's recommeded to retrain the tokenizer model when the 
  data is changed.
  """

  log = get_logger(cfg)

  try:
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(files=list_files_paths(RP.DATA), 
                    vocab_size=cfg.vocab_size, 
                    min_frequency=cfg.min_freq, 
                    special_tokens=['<pad>', '<unk>', '<bos>', '<eos>']
    )
    
    os.makedirs(RP.TOKENIZER_MODEL, exist_ok=True)

    tokenizer.save_model(RP.TOKENIZER_MODEL)

    log.debug(f'Tokenizer model saved to "' + RP.TOKENIZER_MODEL + '"')
  
  except Exception as e:
    log.error(f'Failed to train tokenizer.', exc_info=True)

    raise e