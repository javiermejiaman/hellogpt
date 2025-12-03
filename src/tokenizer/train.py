import src.config as C
from tokenizers import ByteLevelBPETokenizer
from src.utils.file_utils import list_files_paths

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=list_files_paths(C.DATA_PATH), 
                vocab_size=C.VOCAB_SIZE, 
                min_frequency=C.MIN_FREQ, 
                special_tokens=['<pad>', '<unk>', '<bos>', '<eos>']
)

tokenizer.save_model(C.TOKENIZER_MODEL_PATH)
