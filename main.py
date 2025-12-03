import sys
import src.config as C
from src.inference import generate
from src.environment import is_tokenizer_model_available, is_model_available
from src.tokenizer.train import train_tokenizer_model
from src.train.train import train_model

if C.SHOW_BANNER:
  print(r"""
   _    _      _ _        _____ _____ _______ 
  | |  | |    | | |      / ____|  __ \__   __|
  | |__| | ___| | | ___ | |  __| |__) | | |   
  |  __  |/ _ \ | |/ _ \| | |_ |  ___/  | |   
  | |  | |  __/ | | (_) | |__| | |      | |   
  |_|  |_|\___|_|_|\___/ \_____|_|      |_|   
                                              
  """)

if not is_tokenizer_model_available():
  print('Tokenizer model not found, do you want to train one? ')
  should_train = input('Press enter to continue, X to exit: ')

  if should_train.lower() != 'x':
    train_tokenizer_model()
  else:
    sys.exit(1)

if not is_model_available():
  print('The model "' + C.MODEL_NAME + '" is not trained, do you want to train it?')
  should_train = input('Press enter to continue, X to exit: ')
  
  if should_train.lower() != 'x':
    train_model()
  else:
    sys.exit(1)

continue_program = True

while(continue_program):
  prompt = input('### Enter a prompt, or exit to quit: ')
  if prompt == 'exit':
    break

  for token in generate(prompt):
    print(token, end='', flush=True)
  
  print('\n\n-------------------------\n\n')