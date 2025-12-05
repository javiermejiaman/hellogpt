import os

def list_file_paths(dir):
  """List file paths.

  Args:
    dir (str): Target directory to list.
  
  Returns:
    list: List of file paths inside the directory.
  """

  if not os.path.isdir(dir):
    return []
  
  return [os.path.join(dir, f)
          for f in os.listdir(dir) 
          if os.path.isfile(os.path.join(dir, f))
  ]
