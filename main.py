import sys
import src.config as C
from src.inference import generate
from src.environment import is_tokenizer_model_available, is_model_available
from src.tokenizer.train import train_tokenizer_model
from src.train.train import Trainer

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
  print('\n\nTokenizer model not found, do you want to train one? ')
  should_train = input('\nPress enter to continue, X to exit: ')

  if should_train.lower() != 'x':
    train_tokenizer_model()
  else:
    sys.exit(1)

if not is_model_available():
  print('\n\nThe model "' + C.MODEL_NAME + '" is not trained, do you want to train it?')
  should_train = input('\nPress enter to continue, X to exit: ')
  
  if should_train.lower() != 'x':
    trainer = Trainer()

    print(f'\n\n📖 TRAINING')

    print(f'\nTraining configuration:')
    print(f'Target number of epochs: {C.EPOCHS}')
    print(f'Batch size: {C.BATCH_SIZE}')
    print(f'Learning rate: {C.LEARNING_RATE}')
    print(f'Dropout: {C.DROPOUT * 100}%')
    print(f'Gradient clipping threshold: {C.GRAD_CLIP}')

    print(f'\nTraining insights:')
    print(f'Number of total samples: {trainer.total_samples}')
    print(f'Number of batches: {trainer.num_batches}')
    print(f'Number of training samples: {trainer.train_size}')
    print(f'Number of validation samples: {trainer.valid_size}')

    print(f'\nEpoch\t\tValidation loss\t\tTraining loss')

    for epoch, valid_loss, train_loss in trainer.train_model():
      print(f'{epoch}\t\t{valid_loss:.6f}\t\t{train_loss:.6f}')

  else:
    sys.exit(1)

continue_program = True

while(continue_program):
  prompt = input('\n\nEnter prompt, X to exit: ')
  print('')

  if prompt.lower() == 'x':
    break

  for token in generate(prompt):
    print(token, end='', flush=True)
  
  print('')