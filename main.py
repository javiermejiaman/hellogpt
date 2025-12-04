import sys
from src.config import Config
from src.inference import Inference
from src.environment import is_tokenizer_model_available, is_model_available
from src.tokenizer.train import train_tokenizer_model
from src.train.train import Trainer

cfg = Config()

if cfg.show_banner:
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
    train_tokenizer_model(cfg)
  else:
    sys.exit(1)

if not is_model_available(cfg):
  print('\n\nThe model "' + cfg.model_name + '" is not trained, do you want to train it?')
  should_train = input('\nPress enter to continue, X to exit: ')
  
  if should_train.lower() != 'x':
    trainer = Trainer(cfg)

    print(f'\n\n📖 TRAINING')

    print(f'\nTraining configuration:')
    print(f'Target number of epochs: {cfg.epochs}')
    print(f'Batch size: {cfg.batch_size}')
    print(f'Learning rate: {cfg.learning_rate}')
    print(f'Dropout: {cfg.dropout * 100}%')
    print(f'Gradient clipping threshold: {cfg.grad_clip}')

    print(f'\nTraining insights:')
    print(f'Number of total samples: {trainer.total_samples:,}')
    print(f'Number of batches: {trainer.num_batches:,}')
    print(f'Number of training samples: {trainer.train_size:,}')
    print(f'Number of validation samples: {trainer.valid_size:,}')

    print(f'\nEpoch\t\tValidation loss\t\tTraining loss')

    for epoch, valid_loss, train_loss in trainer.train_model():
      print(f'{epoch}\t\t{valid_loss:.6f}\t\t{train_loss:.6f}')

  else:
    sys.exit(1)

inference = Inference(cfg)

continue_program = True
while(continue_program):
  prompt = input('\n\nEnter prompt, X to exit: ')
  print('')

  if prompt.lower() == 'x':
    break

  for token in inference.generate(prompt):
    print(token, end='', flush=True)
  
  print('')