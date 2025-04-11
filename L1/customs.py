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


class Linear4(nn.Module):
    def __init__(self, in_size, out_size, bias=True, *args, **kwargs):

        super().__init__(*args, *kwargs)

        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias

        self.lin1 = nn.Linear(self.in_size, self.out_size)
        self.lin2 = nn.Linear(self.out_size, self.out_size)
        self.lin3 = nn.Linear(self.out_size, self.out_size)
        self.lin4 = nn.Linear(self.out_size, self.out_size)

    def forward(self, x):
        h = self.lin1(x)
        h = self.lin2(h)
        h = self.lin3(h)
        h = self.lin4(h)

        return h
