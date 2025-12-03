import numpy as np
import torch
from torch.utils.data import DataLoader
from src.config import Config
from torch.nn.utils import clip_grad_norm_
from src.utils.model_utils import ModelUtils
from torch.optim import AdamW
from src.environment import get_device

class TrainUtils():

  def __init__(self, cfg: Config):
    self.cfg = cfg
    self.model_utils = ModelUtils(cfg)
    
  def get_model(self):
    """Get latest model and optimizer.

    Returns:
      Model: Instance of the latest version of the model.
      AdamW: Model optimizer.
    """

    model = self.model_utils.load_model()
    
    return model, AdamW(model.parameters(), lr=self.cfg.learning_rate)

  def get_data(self, train_ds, valid_ds):
    """Gets the training and validation data loaders.
    
    Args:
      train_ds (TextDataset): Training dataset.
      valid_ds (TextDataset): Validation dataset.
    
    Returns:
      (DataLoader, DataLoader): Training and validation data loaders.
    """

    return (
      DataLoader(train_ds, 
                batch_size=self.cfg.batch_size, 
                shuffle=True),
      DataLoader(valid_ds, 
                batch_size=self.cfg.batch_size * 2, 
                shuffle=True)
    )

  def _loss_batch(self, model, loss_func, xb, yb, opt=None):
    """Calculate loss for a batch of sequences.
    
    Args:
      model (Model): Model instance.
      loss_func (function): Loss function to apply.
      xb (Tensor): shape (B, S) - Batch of input token sequences
                  to pass through the model.
      yb (Tensor): shape (B, S) - Batch of target token sequences.
    
    Returns:
      (float, int): Tuple with the loss and batch size.
    """

    logits = model(xb)
    loss = loss_func(logits.view(-1, logits.size(-1)), yb.view(-1))

    if opt is not None:
      loss.backward()
      clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
      opt.step()
      opt.zero_grad()

    return loss.item(), len(xb)

  def fit(self, epochs, model, loss_func, opt, train_dl, valid_dl):
    """Trains and validates the model.
    
    Args:
      epochs (int): Number of epochs to train.
      model (Model): Model instance.
      loss_func (function): Loss function to apply.
      opt (AdamW): Optimizer to apply.
      train_dl (DataLoader): Training data loader.
      valid_dl (DataLoader): Validation data loader.
    """

    for epoch in range(1, epochs + 1):
      total_loss = 0
      total_samples = 0

      model.train()
      for xb, yb in train_dl:
        xb, yb = xb.to(get_device()), yb.to(get_device())
        losses, batch_size = self._loss_batch(model, loss_func, xb, yb, opt)
        total_loss += losses * batch_size
        total_samples += batch_size
      
      model.eval()
      with torch.no_grad():
        losses, batch_size = zip(
          *[self._loss_batch(model, loss_func, xb.to(get_device()), yb.to(get_device())) 
            for xb, yb in valid_dl]
        )
      valid_loss = np.sum(np.multiply(losses, batch_size)) / np.sum(batch_size)
      train_loss = total_loss / total_samples

      self.model_utils.save_model(model, opt, epoch, train_loss)

      yield (epoch, valid_loss, train_loss)
