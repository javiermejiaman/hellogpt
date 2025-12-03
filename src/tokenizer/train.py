import src.config as C
from tokenizers import ByteLevelBPETokenizer
from src.utils.file_utils import list_files_paths

def train_tokenizer_model():
  """Trains the tokenizer model.
  
  The source for this tokenizer model is the training data.
  It's recommeded to retrain the tokenizer model when the 
  data is changed.
  """

  tokenizer = ByteLevelBPETokenizer()

  tokenizer.train(files=list_files_paths(C.DATA_PATH), 
                  vocab_size=C.VOCAB_SIZE, 
                  min_frequency=C.MIN_FREQ, 
                  special_tokens=['<pad>', '<unk>', '<bos>', '<eos>']
  )

  tokenizer.save_model(C.TOKENIZER_MODEL_PATH)
