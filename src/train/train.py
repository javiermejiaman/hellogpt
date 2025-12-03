from torch.utils.data import random_split
from src.config import Config
from src.train.dataset import TextDataset
import torch.nn.functional as F
from src.train.utils import TrainUtils

class Trainer:

  def __init__(self, cfg: Config):
    self._train_utils = TrainUtils(cfg)
    self._epochs = cfg.epochs
    self._loss_func = F.cross_entropy

    self._dataset = TextDataset(cfg)

    self.train_size = int(cfg.train_to_valid_ratio * len(self._dataset))
    self.valid_size = len(self._dataset) - self.train_size
    self.total_samples = self.train_size + self.valid_size

    self._train_ds, self._valid_ds = random_split(
      self._dataset, [self.train_size, self.valid_size])
    self._train_dl, self._valid_dl = self._train_utils.get_data(
      self._train_ds, self._valid_ds)
    self.num_batches = len(self._train_dl) + len(self._valid_dl)

  def train_model(self, epochs: int=None):
    """Trains the model.
    
    Args:
      epochs (int): Number of epochs to train.
    """

    epochs = self._epochs if epochs is None else epochs

    model, opt = self._train_utils.get_model()
    yield from self._train_utils.fit(epochs, model, 
      self._loss_func, opt, 
      self._train_dl, self._valid_dl)
