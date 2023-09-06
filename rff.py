# Copied from: https://github.com/ndahlquist/pytorch-fourier-feature-networks/blob/master/fourier_feature_transform.py
import torch
from math import pi

class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_feats, mapping_size=128, scale=10):
        super().__init__()

        self._num_input_feats = num_input_feats
        self._mapping_size = mapping_size
        self._B = torch.randn((num_input_feats, mapping_size)) * scale

    def forward(self, x):
        batches, seq_len, num_feats = x.shape
        x = x.reshape(batches * seq_len, num_feats)
        x = x @ self._B.to(x.device)
        x = x.view(batches, seq_len, self._mapping_size)
        x = 2 * pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    
if __name__ == "__main__":
    
    batch_size, seq_length, num_input_feats =  16, 50, 4
    x = torch.randn((batch_size, seq_length, num_input_feats))

    # Create the feature transform.
    feature_transform = GaussianFourierFeatureTransform(num_input_feats=num_input_feats, mapping_size=128, scale=2)

    # Apply the feature transform.
    x = feature_transform(x)

    print(x.shape)