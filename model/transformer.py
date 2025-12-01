import torch
import torch.nn as nn
import config as C

class TransformerBlock(nn.Module):
  """Transformer block.

  General architecture:

    ATTENTION BLOCK
      CM -> Causal mask
      LN -> Layer normalization
      AT -> Multi-head attention
      RD -> Residual connections + dropout
      
    FEED-FORWARD BLOCK
      LN -> Layer normalization
      FF -> Feed-forward sub-layers
      RD -> Residual connections + dropout

  Description:
    CM) Prevents tokens from attending to future positions.
    LN) Stabilizes training and prevent gradient spikes.
    AT) Enriches embeddings with contextual information.
    FF) Neural network applied independently to each token 
        embedding to produce a refined representation.
    RD) Prevents gradient vanishing by letting it flow 
        through the skip path. 
        Drop out is also applied to disable activations 
        and prevent overfitting.
  """
  
  def __init__(self):
    super().__init__()
    self.attn = nn.MultiheadAttention(C.D_MODEL, C.NUM_HEADS, C.DROPOUT, batch_first=True)
    self.ln_attn = nn.LayerNorm(C.D_MODEL)
    self.ln_ff = nn.LayerNorm(C.D_MODEL)
    self.dropout = nn.Dropout(C.DROPOUT)
    self.ff = nn.Sequential(
      nn.Linear(C.D_MODEL, C.D_FF),
      nn.GELU(),
      nn.Linear(C.D_FF, C.D_MODEL)
    )
  
  def forward(self, batch):
    """Forward passes the input batch through the transformer.

    Args:
      batch (Tensor): shape (B, S, D) - Batch of sequences 
      of token embeddings.
    
    Returns:
      Tensor: shape (B, S, D) - Updated representations after 
      self-attention and feed-forward pass.
    """
    batch_seq_len=batch.size()[1]

    self.causal_mask = torch.triu(
      torch.ones(batch_seq_len, batch_seq_len, device=C.DEVICE, dtype=torch.bool),
      diagonal=1
    )
    
    batch = self.ln_attn(batch)
    attn = self.attn(
      batch, batch, batch,
      attn_mask=self.causal_mask,
      need_weights=False
    )[0]
    batch = batch + self.dropout(attn)
    
    batch = self.ln_ff(batch)
    ff = self.ff(batch)
    batch = batch + self.dropout(ff)

    return batch