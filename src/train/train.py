from torch.utils.data import random_split
import src.config as C
from train.dataset import TextDataset
import torch.nn.functional as F

from train.utils import fit, get_data, get_model

loss_func = F.cross_entropy

dataset = TextDataset()

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_ds, valid_ds = random_split(dataset, [train_size, val_size])

train_dl, valid_dl = get_data(train_ds, valid_ds)
model, opt = get_model()
fit(C.EPOCHS, model, loss_func, opt, train_dl, valid_dl)
