from torch.utils.data import random_split
from src.config import Config
from src.train.dataset import TextDataset
import torch.nn.functional as F
from src.train.utils import fit, get_data, get_model

class Trainer:

  def __init__(self, cfg: Config):
    self.epochs = cfg.epochs
    self._loss_func = F.cross_entropy

    self.dataset = TextDataset()

    self.train_size = int(cfg.train_to_valid_ratio * len(self.dataset))
    self.valid_size = len(self.dataset) - self.train_size
    self.total_samples = self.train_size + self.valid_size

    self.train_ds, self.valid_ds = random_split(
      self.dataset, [self.train_size, self.valid_size])
    self.train_dl, self.valid_dl = get_data(self.train_ds, self.valid_ds)
    self.num_batches = len(self.train_dl) + len(self.valid_dl)

  def train_model(self, epochs: int=None):
    """Trains the model.
    
    Args:
      epochs (int): Number of epochs to train.
    """

    epochs = self.epochs if epochs is None else epochs

    model, opt = get_model()
    yield from fit(epochs, model, self._loss_func, opt, self.train_dl, self.valid_dl)
