from src.inference import generate

print(r"""
  _    _      _ _        _____ _____ _______ 
 | |  | |    | | |      / ____|  __ \__   __|
 | |__| | ___| | | ___ | |  __| |__) | | |   
 |  __  |/ _ \ | |/ _ \| | |_ |  ___/  | |   
 | |  | |  __/ | | (_) | |__| | |      | |   
 |_|  |_|\___|_|_|\___/ \_____|_|      |_|   
                                             
                                             
      """)

enter_prompt = True

while(enter_prompt):
  prompt = input('### Enter a prompt, or exit to quit: ')
  if prompt == 'exit':
    break

  for token in generate(prompt):
    print(token, end='', flush=True)
  
  print('\n\n-------------------------\n\n')