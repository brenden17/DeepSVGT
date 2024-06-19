"""

https://github.com/CUN-bjy/lstm-vae-torch/blob/main/src/models.py
"""
import math
from dataclasses import dataclass


import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import BASE_SIZE


torch.manual_seed(0)

@dataclass
class Parameters:
    n_layers: int = 2
    hidden_dim: int = 10

    latent_dims: int = 2


class EncoderLayers(nn.Module):
    def __init__(self, seq_size, params):
        super(EncoderLayers, self).__init__()

        self.n_layers = params.n_layers
        self.hidden_dim = params.hidden_dim
        self.latent_dims = params.latent_dims


        # self.rnn = nn.RNN(BASE_SIZE, self.hidden_dim, self.n_layers)
        self.rnn = nn.LSTM(BASE_SIZE, self.hidden_dim, self.n_layers)

        self.linear1 = nn.Linear(self.conv_1_out_size*self.conv_1_out_channels, 512)

        self.linear2 = nn.Linear(512, self.latent_dims)
        self.linear3 = nn.Linear(512, self.latent_dims)


    def forward(self, x):
        batch_size, seq_size, bases = x.shape
        # x = x.reshape(batch, bases, seq_size)

        hidden_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        hidden_outputs, hidden_state = self.rnn(x, hidden_state)

        # hidden state : [layer_size, batch_size, hidden_size]
        # print(hidden_state.shape)
        hidden_outputs = hidden_outputs.contiguous().view(-1, self.hidden_dim)
        hidden_state = hidden_state.reshape((batch_size, self.hidden_dim))
        # print('hidden_state.shape')
        # print(hidden_state.shape)

        # x = torch.flatten(x, start_dim=1)


        x = F.relu(self.linear1(hidden_outputs))

        mu = self.linear2(x)
        log_var = self.linear3(x)

        return mu, log_var, hidden_state


class DecoderLayers(nn.Module):
    def __init__(self, seq_size, params):
        super(DecoderLayers, self).__init__()
        
        self.n_layers = params.n_layers
        self.hidden_dim = params.hidden_dim
        self.latent_dims = params.latent_dims

        # self.rnn = nn.RNN(BASE_SIZE, self.hidden_dim, self.n_layers)
        self.rnn = nn.LSTM(BASE_SIZE, self.hidden_dim, self.n_layers)

        self.linear1 = nn.Linear(self.latent_dims, 512)
        self.linear2 = nn.Linear(512, self.conv_1_out_size*self.conv_1_out_channels)

    def forward(self, z, hidden_state):
        z = F.relu(self.linear1(z))

        x = torch.sigmoid(self.linear2(z))

        x = x.view(batch_size, -1, self.hidden_dim)

        hidden_outputs, hidden_state = self.rnn(x, hidden_state)


        batch, _ = hidden_outputs.shape
        # x = x.view(batch, self.conv_1_out_channels, self.conv_1_out_size)
        # x = F.relu(self.conv1(x))

        # x = torch.sigmoid(x)

        # batch, bases, seq_size = x.shape
        x = x.reshape(batch, seq_size, bases)

        return hidden_outputs

class RNNAutoencoder(nn.Module):
    def __init__(self, seq_size, latent_dims):
        super(RNNAutoencoder, self).__init__()
        params = Parameters()
        params.latent_dims = latent_dims

        self.encoder = EncoderLayers(seq_size, params)
        self.decoder = DecoderLayers(seq_size, params)

    def forward(self, x):
        mu, log_var, hidden_state = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z, hidden_state)

        return x_hat, mu, log_var

    def loss_fn(self, x_hat, x, mu, log_var):
        MSE = F.binary_cross_entropy(x_hat, x, size_average=False)
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + KLD

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std