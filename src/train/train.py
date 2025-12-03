from torch.utils.data import random_split
import src.config as C
from src.train.dataset import TextDataset
import torch.nn.functional as F
from src.train.utils import fit, get_data, get_model

class Trainer:

  def __init__(self):
    self._loss_func = F.cross_entropy

    self.dataset = TextDataset()

    self.train_size = int(C.TRAIN_TO_VALID_RATIO * len(self.dataset))
    self.valid_size = len(self.dataset) - self.train_size
    self.total_samples = self.train_size + self.valid_size

    self.train_ds, self.valid_ds = random_split(self.dataset, [self.train_size, self.valid_size])
    self.train_dl, self.valid_dl = get_data(self.train_ds, self.valid_ds)
    self.num_batches = len(self.train_dl) + len(self.valid_dl)

  def train_model(self, epochs: int=C.EPOCHS):
    """Trains the model.
    
    Args:
      epochs (int): Number of epochs to train.
    """

    model, opt = get_model()
    yield from fit(epochs, model, self._loss_func, opt, self.train_dl, self.valid_dl)
