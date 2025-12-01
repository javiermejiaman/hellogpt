import torch
import torch.nn as nn
import config as C

class TransformerBlock(nn.Module):
  
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
      
      Args:
        batch (Tensor): shape (B, S, D_MODEL) - Batch of sequences 
        of token embeddings.
      
      Returns:
        Tensor: shape (B, S, D_MODEL) - Updated representations after 
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

class Model(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.token_embedding = nn.Embedding(C.VOCAB_SIZE, C.D_MODEL)
    self.pos_embedding = nn.Embedding(C.SEQ_LEN, C.D_MODEL)

    self.layers = nn.ModuleList([TransformerBlock() for _ in range(C.NUM_LAYERS)])

    self.ln_head = nn.LayerNorm(C.D_MODEL)
    self.head = nn.Linear(C.D_MODEL, C.VOCAB_SIZE, bias=False)

    self.dropout = nn.Dropout(C.DROPOUT)

  def forward(self, batch):
    """Forward passes the input batch through the model.
    
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
      
      Args:
        batch (Tensor): shape (B, S) - Batch of token sequences.
      
      Returns:
        Tensor: shape (B, S, VOCAB_SIZE) - Logits for each token in the batch.
    """
    positions = torch.arange(batch.size()[1], device=C.DEVICE).unsqueeze(0)

    batch = self.token_embedding(batch) + self.pos_embedding(positions)
    batch = self.dropout(batch)

    for layer in self.layers:
      batch = layer(batch)

    batch = self.ln_head(batch)
    
    return self.head(batch)
