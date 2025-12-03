import torch
import torch.nn as nn
from src.config import Config
from src.environment import get_device

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
  
  def __init__(self, cfg: Config):
    super().__init__()
    self.attn = nn.MultiheadAttention(cfg.d_model, 
                                      cfg.num_heads, 
                                      cfg.dropout, 
                                      batch_first=True)
    self.ln_attn = nn.LayerNorm(cfg.d_model)
    self.ln_ffn = nn.LayerNorm(cfg.d_model)
    self.dropout = nn.Dropout(cfg.dropout)
    self.ffn = nn.Sequential(
      nn.Linear(cfg.d_model, cfg.d_ffn),
      nn.GELU(),
      nn.Linear(cfg.d_ffn, cfg.d_model)
    )
  
  def get_causal_mask(self, seq_len: int):
    """Creates a boolean causal mask.

    Used to prevents tokens from attending to future positions.
    
    Args:
      seq_len (int): Length of the current sequence.
    
    Returns:
      Tensor: shape (S, S) - Upper triangular boolean mask.
    """
    ones_matrix = torch.ones(seq_len, 
                             seq_len, 
                             device=get_device(), 
                             dtype=torch.bool
    )                                                      # (S, S)

    return torch.triu(ones_matrix, diagonal=1)             # (S, S)
  
  def forward(self, batch):
    """Forward passes the input batch through the transformer.

    Args:
      batch (Tensor): shape (B, S, D) - Batch of sequences 
      of token embeddings.
    
    Returns:
      Tensor: shape (B, S, D) - Updated representations after 
      self-attention and feed-forward pass.
    """
    causal_mask = self.get_causal_mask(batch.size()[1])
    
    batch = self.ln_attn(batch)                            # (B, S, D)
    attn = self.attn(
      batch, batch, batch,
      attn_mask=causal_mask,
      need_weights=False
    )[0]                                                   # (B, S, D)
    batch = batch + self.dropout(attn)                     # (B, S, D)
    
    batch = self.ln_ffn(batch)                             # (B, S, D)
    ffn = self.ffn(batch)                                  # (B, S, D)
    batch = batch + self.dropout(ffn)                      # (B, S, D)

    return batch                                           # (B, S, D)