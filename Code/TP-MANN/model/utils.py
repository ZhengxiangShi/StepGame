import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, equation, in_features, hidden_size, out_size):
        super(MLP, self).__init__()
        self.equation = equation
        # Layers
        # 1
        self.W1 = nn.Parameter(torch.zeros(in_features, hidden_size))
        nn.init.xavier_uniform_(self.W1.data)
        self.b1 = nn.Parameter(torch.zeros(hidden_size))
        # 2
        self.W2 = nn.Parameter(torch.zeros(hidden_size, out_size))
        nn.init.xavier_uniform_(self.W2.data)
        self.b2 = nn.Parameter(torch.zeros(out_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(torch.einsum(self.equation, x, self.W1) + self.b1)
        out = torch.tanh(torch.einsum(self.equation, hidden, self.W2) + self.b2)
        return out


class OptionalLayer(nn.Module):
    def __init__(self, layer: nn.Module, active: bool = False):
        super(OptionalLayer, self).__init__()
        self.layer = layer
        self.active = active
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active:
            return self.layer(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = (x - mu).pow(2).mean(-1, keepdim=True)
        normalized = (x - mu) / (torch.sqrt(sigma + self.eps))
        return normalized * self.gain + self.bias
