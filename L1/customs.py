import torch 
import torch.nn as nn


class CustomLinear(nn.Module):
    """
        Custom Fully-Connected Layer.
    """
    def __init__(self, in_size, out_size, bias=True, *args, **kwargs):

        super().__init__(*args, *kwargs)

        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias

        # Define your custom parameters (weights and bias)
        self.W = nn.Parameter(torch.Tensor(out_size, in_size))
        torch.nn.init.xavier_uniform_(self.W)
        
        if self.bias:
            self.bias = nn.Parameter(torch.randn(out_size))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.bias is not None:
            return torch.matmul(x, self.W.t()) + self.bias
        else:
            return torch.matmul(x, self.W.t())