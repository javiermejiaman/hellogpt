from inference import generate

enter_prompt = True

while(enter_prompt):
  prompt = input('### Enter a prompt, or exit to quit: ')
  if prompt == 'exit':
    break

  for token in generate(prompt):
    print(token, end='', flush=True)
  
  print('\n\n-------------------------\n\n')