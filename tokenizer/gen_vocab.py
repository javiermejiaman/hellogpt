import os
import config as C
from tokenizers import ByteLevelBPETokenizer

file_paths = [os.path.join(C.DATA_PATH, f)
              for f in os.listdir(C.DATA_PATH) 
              if os.path.isfile(os.path.join(C.DATA_PATH, f))
]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=file_paths, 
                vocab_size=C.VOCAB_SIZE, 
                min_frequency=C.MIN_FREQ, 
                special_tokens=['<pad>', '<unk>', '<bos>', '<eos>']
)

tokenizer.save_model(C.TOKENIZER_MODEL_PATH)
