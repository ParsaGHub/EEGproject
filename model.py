import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """
    A single FiLM-conditioned hidden layer.
    Computes h = gamma(c) * phi(W*h_prev + b) + beta(c)
    """
    def __init__(self, in_dim, out_dim, cond_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.gamma_net = nn.Linear(cond_dim, out_dim)
        self.beta_net = nn.Linear(cond_dim, out_dim)
        self.activation = nn.GELU()

    def forward(self, h, c):
        z = self.activation(self.linear(h))
        gamma = self.gamma_net(c)
        beta = self.beta_net(c)
        return gamma * z + beta


class FiLMMLP(nn.Module):
    """
    FiLM-conditioned MLP for EEG dipole localization.

    Inputs:
        V  : (batch, n_electrodes) - scalp voltages
        c  : (batch, 8)            - head parameters [sigma (4), radii (4)]

    Output:
        out: (batch, 6)            - predicted [p (3), r0 (3)]
    """
    def __init__(self, n_electrodes=64, cond_dim=8, hidden_dim=256, n_layers=4):
        super().__init__()

        self.layers = nn.ModuleList()

        # first layer: input is voltages
        self.layers.append(FiLMLayer(n_electrodes, hidden_dim, cond_dim))

        # subsequent layers
        for _ in range(n_layers - 1):
            self.layers.append(FiLMLayer(hidden_dim, hidden_dim, cond_dim))

        # linear readout
        self.readout = nn.Linear(hidden_dim, 6)

    def forward(self, V, c):
        h = V
        for layer in self.layers:
            h = layer(h, c)
        return self.readout(h)