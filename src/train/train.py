import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from model.model import Model
import src.config as C
from train.dataset import TextDataset
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from utils.file_utils import get_model_latest_serial, get_model_path

def get_model():
  model = Model().to(C.DEVICE)
  model.load_state_dict(torch.load(get_model_path(get_model_latest_serial()), 
                                  map_location=C.DEVICE))
  model.eval()
  return model, torch.optim.AdamW(model.parameters(), lr=C.LEARNING_RATE)

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
    model.train()
    for xb, yb in train_dl:
      xb, yb = xb.to(C.DEVICE), yb.to(C.DEVICE)
      loss_batch(model, loss_func, xb, yb, opt)

    model.eval()
    with torch.no_grad():
      losses, nums = zip(
        *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
      )
    valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

    print(epoch, valid_loss)

    torch.save(model.state_dict(), get_model_path(get_model_latest_serial() + 1))

loss_func = F.cross_entropy

dataset = TextDataset()

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_ds, valid_ds = random_split(dataset, [train_size, val_size])

train_dl, valid_dl = get_data(train_ds, valid_ds)
model, opt = get_model()
fit(C.EPOCHS, model, loss_func, opt, train_dl, valid_dl)
