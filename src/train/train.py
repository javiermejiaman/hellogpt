from torch.utils.data import random_split
import src.config as C
from src.train.dataset import TextDataset
import torch.nn.functional as F
from src.train.utils import fit, get_data, get_model

def train_model(epochs: int=C.EPOCHS):
  """Trains the model.
  
  Args:
    epochs (int): Number of epochs to train.
  """

  loss_func = F.cross_entropy

  dataset = TextDataset()

  train_size = int(C.TRAIN_TO_VALID_RATIO * len(dataset))
  valid_size = len(dataset) - train_size

  train_ds, valid_ds = random_split(dataset, [train_size, valid_size])

  train_dl, valid_dl = get_data(train_ds, valid_ds)
  model, opt = get_model()
  yield from fit(epochs, model, loss_func, opt, train_dl, valid_dl)
