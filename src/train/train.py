import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import src.config as C
from train.dataset import TextDataset
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from utils.file_utils import load_model, save_model
from torch.optim import AdamW

def get_model():
  model = load_model()
  
  return model, AdamW(model.parameters(), lr=C.LEARNING_RATE)

def get_data(train_ds, valid_ds):
  return (
    DataLoader(train_ds, 
               batch_size=C.BATCH_SIZE, 
               shuffle=True),
    DataLoader(valid_ds, 
               batch_size=C.BATCH_SIZE * 2, 
               shuffle=True)
  )

def loss_batch(model, loss_func, xb, yb, opt=None):
  logits = model(xb)
  loss = loss_func(logits.view(-1, logits.size(-1)), yb.view(-1))

  if opt is not None:
    loss.backward()
    clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
    opt.step()
    opt.zero_grad()

  return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):

  for epoch in range(epochs):
    total_loss = 0
    total_samples = 0

    model.train()
    for xb, yb in train_dl:
      xb, yb = xb.to(C.DEVICE), yb.to(C.DEVICE)
      losses, batch_size = loss_batch(model, loss_func, xb, yb, opt)
      total_loss += losses * batch_size
      total_samples += batch_size
    
    model.eval()
    with torch.no_grad():
      losses, batch_size = zip(
        *[loss_batch(model, loss_func, xb.to(C.DEVICE), yb.to(C.DEVICE)) 
          for xb, yb in valid_dl]
      )
    valid_loss = np.sum(np.multiply(losses, batch_size)) / np.sum(batch_size)

    print(epoch, valid_loss)

    save_model(model, opt, epoch, total_loss / total_samples)

loss_func = F.cross_entropy

dataset = TextDataset()

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_ds, valid_ds = random_split(dataset, [train_size, val_size])

train_dl, valid_dl = get_data(train_ds, valid_ds)
model, opt = get_model()
fit(C.EPOCHS, model, loss_func, opt, train_dl, valid_dl)
