
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativePositionalEncoding(nn.Module):
  def __init__(self, d_model, max_relative_pos=256):
    super(RelativePositionalEncoding, self).__init__()
    self.d_model = d_model
    self.max_relative_pos = max_relative_pos
    
    # Create relative positional embeddings table
    self.relative_pos_embed = nn.Embedding(2 * max_relative_pos + 1, d_model)
    self._init_relative_pos_embed()
    
  def _init_relative_pos_embed(self):
    # Initialization of relative positional embeddings
    nn.init.normal_(self.relative_pos_embed.weight, mean=0, std=0.02)
      
  def forward(self, q, k):
    batch_size, seq_len, _ = q.size()

    relative_positions = torch.arange(-seq_len + 1, 1).unsqueeze(0).repeat(batch_size, seq_len, 1).to(q.device)
    relative_pos_enc = self.relative_pos_embed(relative_positions + self.max_relative_pos)
    
    print("Positional shapes")
    print(q.shape)
    print(k.shape)
    print(relative_pos_enc.shape)
    exit()
    
    return q + relative_pos_enc, k + relative_pos_enc
  

class RelTransformerEncoder(nn.Module):

  def __init__(self,
               ) -> None:
    super().__init__()