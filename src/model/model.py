import torch
import torch.nn as nn
import src.config as C
import transformer

class Model(nn.Module):
  """HelloGPT model.

  General architecture:
    1 -> Embeddings lookup and addition
    2 -> Dropout
    3 -> -> -> Transformers
    4 -> Layer normalization
    5 -> Head layer

  Description:
    1) Encodes tokens and their position in the sequence.
    2) Randomly disables activations to prevent model overfitting.
    3) Applies self attention and feed-forward networks to enrich 
       embeddings with contextual information.
    4) Stabilizes training and prevent gradient spikes.
    5) Linear projection mapping each token embedding to the 
       vocabulaby logits.
  """
  
  def __init__(self):
    super().__init__()
    self.token_embedding = nn.Embedding(C.VOCAB_SIZE, C.D_MODEL)
    self.pos_embedding = nn.Embedding(C.MAX_SEQ_LEN, C.D_MODEL)

    self.layers = nn.ModuleList([transformer.TransformerBlock() for _ in range(C.NUM_LAYERS)])

    self.ln_head = nn.LayerNorm(C.D_MODEL)
    self.head = nn.Linear(C.D_MODEL, C.VOCAB_SIZE, bias=False)

    self.dropout = nn.Dropout(C.DROPOUT)

  def forward(self, batch):
    """Forward passes the input batch through the model.
    
    Args:
      batch (Tensor): shape (B, S) - Batch of token sequences.
    
    Returns:
      Tensor: shape (B, S, V) - Logits for each token in the batch.
    """
    positions = torch.arange(batch.size()[1], 
                             device=C.DEVICE
    ).unsqueeze(0)                               # (1, S)

    batch = (
      self.token_embedding(batch) 
      + self.pos_embedding(positions)
    )                                            # (B, S, D)
    
    batch = self.dropout(batch)                  # (B, S, D)

    for layer in self.layers:
      batch = layer(batch)                       # (B, S, D)

    batch = self.ln_head(batch)                  # (B, S, D)
    
    return self.head(batch)                      # (B, S, V)
