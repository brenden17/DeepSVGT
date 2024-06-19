"""
https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/vae/vq_vae.py
https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb
https://github.com/rosinality/vq-vae-2-pytorch
https://github.com/FernandoLpz/Text-Classification-CNN-PyTorch/tree/master/src/parameters
"""
import math
from dataclasses import dataclass


import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import EXTEND_BASE_SIZE

random_seed = 1
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@dataclass
class Parameters:
    conv_1_out_channels: int = 10
    conv_1_kernel_size: int = 10
    conv_1_stride: int = 1

    middle_dims: int = 32
    latent_dims: int = 2


def calculate_output_size_conv1d(seq_size, kernel_size, stride):
    '''
    source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    '''
    out_conv = ((seq_size - 1 * (kernel_size - 1) - 1) / stride) + 1
    out_conv = math.floor(out_conv)

    return out_conv

class EncoderLayers(nn.Module):
    def __init__(self, seq_size, params):
        super(EncoderLayers, self).__init__()

        self.seq_size = seq_size

        self.conv_1_out_channels = params.conv_1_out_channels
        self.conv_1_kernel_size = params.conv_1_kernel_size
        self.conv_1_stride = params.conv_1_stride

        self.middle_dims = params.middle_dims
        self.latent_dims = params.latent_dims

        self.conv_1 = nn.Conv1d(EXTEND_BASE_SIZE, self.conv_1_out_channels, self.conv_1_kernel_size, self.conv_1_stride)
        self.conv_1_out_size = calculate_output_size_conv1d(self.seq_size, self.conv_1_kernel_size, self.conv_1_stride)

        self.linear1 = nn.Linear(self.conv_1_out_size*self.conv_1_out_channels, self.middle_dims)

        self.linear2 = nn.Linear(self.middle_dims, self.latent_dims)
        self.linear3 = nn.Linear(self.middle_dims, self.latent_dims)


    def forward(self, x):
        batch, seq_size, bases = x.shape
        x = x.reshape(batch, bases, seq_size)
        x = self.conv_1(x)
        x = torch.relu(x)

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.linear1(x))

        mu = self.linear2(x)
        log_var = self.linear3(x)

        return mu, log_var


class DecoderLayers(nn.Module):
    def __init__(self, seq_size, params):
        super(DecoderLayers, self).__init__()

        self.seq_size = seq_size

        self.conv_1_out_channels = params.conv_1_out_channels
        self.conv_1_kernel_size = params.conv_1_kernel_size
        self.conv_1_stride = params.conv_1_stride
        
        self.middle_dims = params.middle_dims
        self.latent_dims = params.latent_dims

        self.conv_1_out_size = calculate_output_size_conv1d(self.seq_size, self.conv_1_kernel_size, self.conv_1_stride)
        self.conv1 = nn.ConvTranspose1d(self.conv_1_out_channels, EXTEND_BASE_SIZE, self.conv_1_kernel_size, self.conv_1_stride)

        self.linear1 = nn.Linear(self.latent_dims, self.middle_dims)
        self.linear2 = nn.Linear(self.middle_dims, self.conv_1_out_size*self.conv_1_out_channels)

    def forward(self, z):
        z = F.relu(self.linear1(z))

        x = torch.sigmoid(self.linear2(z))

        batch, _ = x.shape
        x = x.view(batch, self.conv_1_out_channels, self.conv_1_out_size)
        x = F.relu(self.conv1(x))

        x = torch.sigmoid(x)

        batch, bases, seq_size = x.shape
        x = x.reshape(batch, seq_size, bases)

        return x

class CNNAutoencoder(nn.Module):
    def __init__(self, seq_size, latent_dims):
        super(CNNAutoencoder, self).__init__()
        self.set_seed()

        params = Parameters()
        params.latent_dims = latent_dims

        self.encoder = EncoderLayers(seq_size, params)
        self.decoder = DecoderLayers(seq_size, params)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)

        return x_hat, mu, log_var

    def loss_fn(self, x_hat, x, mu, log_var):
        #MSE = F.binary_cross_entropy(x_hat, x, size_average=False)
        MSE = F.binary_cross_entropy(x_hat, x, reduction='sum')
        #MSE = F.binary_cross_entropy(x_hat, x)
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + KLD

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def set_seed(self):
        random_seed = 1
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Encoder2Layers(nn.Module):
    def __init__(self, seq_size, params):
        super(Encoder2Layers, self).__init__()

        self.seq_size = seq_size

        self.conv_1_out_channels = params.conv_1_out_channels
        self.conv_1_kernel_size = params.conv_1_kernel_size
        self.conv_1_stride = params.conv_1_stride
        self.latent_dims = params.latent_dims

        self.conv_1 = nn.Conv1d(EXTEND_BASE_SIZE, self.conv_1_out_channels, self.conv_1_kernel_size, self.conv_1_stride)
        self.conv_1_out_size = calculate_output_size_conv1d(self.seq_size, self.conv_1_kernel_size, self.conv_1_stride)

        self.linear1 = nn.Linear(self.conv_1_out_size*self.conv_1_out_channels, 512)
        self.linear2 = nn.Linear(512, 256)

        self.linear3 = nn.Linear(256, self.latent_dims)
        self.linear4 = nn.Linear(256, self.latent_dims)


    def forward(self, x):
        batch, seq_size, bases = x.shape
        x = x.reshape(batch, bases, seq_size)
        x = self.conv_1(x)
        x = torch.relu(x)

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        mu = self.linear3(x)
        log_var = self.linear4(x)

        return mu, log_var


class Decoder2Layers(nn.Module):
    def __init__(self, seq_size, params):
        super(Decoder2Layers, self).__init__()

        self.seq_size = seq_size

        self.conv_1_out_channels = params.conv_1_out_channels
        self.conv_1_kernel_size = params.conv_1_kernel_size
        self.conv_1_stride = params.conv_1_stride
        self.latent_dims = params.latent_dims

        self.conv_1_out_size = calculate_output_size_conv1d(self.seq_size, self.conv_1_kernel_size, self.conv_1_stride)
        self.conv1 = nn.ConvTranspose1d(self.conv_1_out_channels, EXTEND_BASE_SIZE, self.conv_1_kernel_size, self.conv_1_stride)

        self.linear1 = nn.Linear(self.latent_dims, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, self.conv_1_out_size*self.conv_1_out_channels)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))

        x = torch.sigmoid(self.linear3(z))

        batch, _ = x.shape
        x = x.view(batch, self.conv_1_out_channels, self.conv_1_out_size)
        x = F.relu(self.conv1(x))

        x = torch.sigmoid(x)

        batch, bases, seq_size = x.shape
        x = x.reshape(batch, seq_size, bases)

        return x

class CNN2LayerAutoencoder(nn.Module):
    def __init__(self, seq_size, latent_dims):
        super(CNN2LayerAutoencoder, self).__init__()
        params = Parameters()
        params.latent_dims = latent_dims

        self.encoder = Encoder2Layers(seq_size, params)
        self.decoder = Decoder2Layers(seq_size, params)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)

        return x_hat, mu, log_var

    def loss_fn(self, x_hat, x, mu, log_var):
        #MSE = F.binary_cross_entropy(x_hat, x, size_average=False)
        MSE = F.binary_cross_entropy(x_hat, x, reduction='sum')
        #MSE = F.binary_cross_entropy(x_hat, x)
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + KLD

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std


def train(model, data, device, epochs=100):
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        for x, y in data:
            x = x.to(device)

            opt.zero_grad()

            # mu, log_var = model.encoder(x)
            # z = model.reparameterize(mu, log_var)
            # x_hat = model.decoder(z)
            x_hat, mu, log_var = model(x)
            loss = model.loss_fn(x_hat, x, mu, log_var)

            loss.backward()
            opt.step()
    return model
