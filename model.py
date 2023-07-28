import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
  """
  Implement the PE function.
  """
  def __init__(self, d_model, max_seq_len=50):
    super().__init__()
    self.d_model = d_model

    # create constant 'pe' matrix with values dependant on
    # pos and i
    pe = torch.zeros(max_seq_len, d_model)
    for pos in range(max_seq_len):
      for i in range(0, d_model, 2):
        pe[pos, i] = math.sin(pos / (10000**((2 * i) / d_model)))
        pe[pos, i + 1] = math.cos(pos / (10000**((2 * (i + 1)) / d_model)))
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    # make embeddings relatively larger
    x = x * math.sqrt(self.d_model)
    #add constant to embedding
    seq_len = x.size(1)

    x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
    return x
  
class StrokeNet(nn.Module):

  def __init__(self,
              user_cnt,
              feat_cnt,
              key_cnt,
              key_emb_size,
              dim_ff,
              num_heads,
              num_layers,
              dropout,
              causal_att,
              use_user_emb) -> None:
    super().__init__()

    self.positional_encoding = PositionalEncoding(key_emb_size * 3, 51)

    self.keycode_embedding = nn.Embedding(key_cnt, key_emb_size)
    self.user_embedding = nn.Embedding(user_cnt, key_emb_size * 3)

    self.feat_cnt = feat_cnt
    self.hidden_dim = key_emb_size
    self.dim_ff = dim_ff
    self.num_heads = num_heads
    self.num_layers = num_layers
    self.dropout = dropout
    self.causal_att = causal_att
    self.use_user_emb = use_user_emb

    self.feat_proj = nn.Linear(feat_cnt, key_emb_size, bias=True)
    self.feat_bn = nn.BatchNorm1d(key_emb_size)

    # self.emb_mlp = nn.Sequential(
    #     nn.Linear(key_emb_size * 3, key_emb_size * 3),
    #     nn.ReLU(),
    #     nn.Linear(key_emb_size * 3, key_emb_size * 3),
    # )

    self.trf_cross = nn.TransformerEncoder(nn.TransformerEncoderLayer(
        d_model=key_emb_size * 3,
        dim_feedforward=dim_ff,
        nhead=self.num_heads,
        dropout=self.dropout,
        batch_first=True,
        norm_first=True),
                                           num_layers=self.num_layers)

    self.electra_lin = nn.Linear(key_emb_size * 3, 2, bias=False)

    if self.use_user_emb:
      self.user_lin = nn.Linear(key_emb_size * 3, 2, bias=False)

  def forward(self, b0, b1, feat, mask, user, attn_mask=None):

    feat = self.feat_proj(feat)

    feat = feat.transpose(1, 2)
    feat = self.feat_bn(feat)
    feat = feat.transpose(1, 2)

    b0_emb = self.keycode_embedding(b0)
    b1_emb = self.keycode_embedding(b1)
    
    user_emb = self.user_embedding(user)

    x = torch.cat([b0_emb, b1_emb, feat], dim=-1)

    # append user embedding to the beginning of the sequence
    x = torch.cat([user_emb.unsqueeze(1), x], dim=1)

    # add positional encoding - including user embedding
    x = self.positional_encoding(x)

    if self.causal_att:
      x = self.trf_cross(src=x, mask=attn_mask, src_key_padding_mask=mask, is_causal=True)
    else:
      x = self.trf_cross(src=x, src_key_padding_mask=mask, is_causal=False)

    # remove user embedding
    user_out = x[:, 0]
    x = x[:, 1:]

    x = self.electra_lin(x)

    if self.use_user_emb:
      user_out = self.user_lin(user_out)
      return x, user_out
    else:
      return x, None

class StrokeRotNet(nn.Module):

  def __init__(self,
               ) -> None:
    super().__init__()