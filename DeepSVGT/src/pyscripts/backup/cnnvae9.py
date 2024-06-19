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

from datasets import build_dataloader, kmer, BASE_SIZE

from utils import plot_latent, create_gif_plots, generate_latent, plot_sim_latent
from simulation import generate_two_groups



torch.manual_seed(0)

# # data, _ = build_dataloaders(seq_size=SEQ_SIZE, group_subs_rate=GROUP_SUBS_RATE, subs_rate=SUBS_RATE, kmer=KMER)

# # model = CNNVariationalAutoencoder(INPUT_SIZE, LATENT_DIMS).to(DEVICE)
# # model = train(model, data)


# # # plot_latent2(model, data, DEVICE, title)
# # x, y = generate_latent(model, data, DEVICE)




# # r = metrics_cluster(x, y)

# # r = ', '.join([f'group {item[0]}:{item[1]:.2f}' for item in r])
# # title = f'seq:{SEQ_SIZE}, kmer:{KMER}, epochs:{EPOCHS}, group:{GROUP_SUBS_RATE}, subs:{SUBS_RATE}, \nnote:{NOTE}, {r}'

# # plot_latent3(x, y, title)

@dataclass
class Parameters:
   conv_1_out_channels: int = 10
   conv_1_kernel_size: int = 2
   conv_1_stride: int = 1
   
   latent_dims: int = 2


def calculate_output_size_conv1d(input_size, kernel_size, stride):
    '''
    source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    '''
    # Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
    print(input_size)
    print(kernel_size)
    print(stride)
    print('============================')
    out_conv = ((input_size - 1 * (kernel_size - 1) - 1) / stride) + 1
    out_conv = math.floor(out_conv)

    return out_conv
    # out_pool = ((out_conv_1 - 1 * (kernel_size - 1) - 1) / stride) + 1
    # out_pool = math.floor(out_pool)
    
    # return out_pool
def calculate_output_size_convtrans1d(input_size, kernel_size, stride):
    '''
    source: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
    '''
    # Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
    out_conv = (input_size - 1) * stride + (kernel_size - 1) + 1

    return out_conv
    # out_pool = ((out_conv_1 - 1 * (kernel_size - 1) - 1) / stride) + 1
    # out_pool = math.floor(out_pool)
    
    # return out_pool

class Encoder2Layers(nn.Module):
    def __init__(self, input_size, params):
        super(Encoder2Layers, self).__init__()

        self.input_size = input_size

        self.conv_1_out_channels = params.conv_1_out_channels
        self.conv_1_kernel_size = params.conv_1_kernel_size
        self.conv_1_stride = params.conv_1_stride
        self.latent_dims = params.latent_dims

        self.conv_1 = nn.Conv1d(BASE_SIZE, self.conv_1_out_channels, self.conv_1_kernel_size, self.conv_1_stride)

        self.conv_1_out_size = calculate_output_size_conv1d(self.input_size, self.conv_1_kernel_size, self.conv_1_stride)
        print(f'out_pool_1-------------------{self.conv_1_out_size}')

        self.linear1 = nn.Linear(self.conv_1_out_size*self.conv_1_out_channels, 512)

        self.linear2 = nn.Linear(512, self.latent_dims)
        self.linear3 = nn.Linear(512, self.latent_dims)


    def forward(self, x):
        # x = torch.flatten(x, start_dim=1)
        # print('befone conv 1')
        # print(x.shape)
        batch, seq_size, bases = x.shape
        x = x.reshape(batch, bases, seq_size)
        x = self.conv_1(x)
        x = torch.relu(x)
        # x = self.pool_1(x)

        # print('after conv 1...')
        # print(x.shape)

        # x = x.view(-1,1,319200)
        x = torch.flatten(x, start_dim=1)

        # print('222222')
        # print(x.shape)


        x = F.relu(self.linear1(x))

        mu = self.linear2(x)
        log_var = self.linear3(x)
        return mu, log_var

class Decoder2Layers(nn.Module):
    def __init__(self, input_size, params):
        super(Decoder2Layers, self).__init__()

        self.input_size = input_size

        self.conv_1_out_channels = params.conv_1_out_channels
        self.conv_1_kernel_size = params.conv_1_kernel_size
        self.conv_1_stride = params.conv_1_stride
        self.latent_dims = params.latent_dims

        self.conv_1_out_size = calculate_output_size_conv1d(self.input_size, self.conv_1_kernel_size, self.conv_1_stride)
        # a = calculate_output_size_convtrans1d(self.conv_1_out_size, conv_1_kernel_size, conv_1_stride)
        # print(f'a/.....{a}================')
        self.conv1 = nn.ConvTranspose1d(self.conv_1_out_channels, BASE_SIZE, self.conv_1_kernel_size, self.conv_1_stride)

        self.linear1 = nn.Linear(self.latent_dims, 512)
        self.linear2 = nn.Linear(512, self.conv_1_out_size*self.conv_1_out_channels)

    def forward(self, z):
        print("22222...")
        print(z.shape)
        
        z = F.relu(self.linear1(z))

        print("8...")
        print(z.shape)

        x = torch.sigmoid(self.linear2(z))

        print("9...")
        print(x.shape)
        batch, _ = x.shape

        x = x.view(batch, self.conv_1_out_channels, self.conv_1_out_size)

        print("10...")
        print(x.shape)

        x = F.relu(self.conv1(x))
        print("10...")
        print(x.shape)

        x = torch.sigmoid(x)

        batch, bases, seq_size = x.shape
        x = x.reshape(batch, seq_size, bases)
        print("11...")
        print(x.shape)

        return x

class CNNAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(CNNAutoencoder, self).__init__()
        params = Parameters()
        params.latent_dims = latent_dims

        self.encoder = Encoder2Layers(input_size, params)
        self.decoder = Decoder2Layers(input_size, params)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)

        return x_hat, mu, log_var

    def loss_fn(self, x_hat, x, mu, log_var):
        # x = x.view(x.size(0), -1) 
        MSE = F.binary_cross_entropy(x_hat, x, size_average=False)
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

SEQ_SIZE = 100
KMER_SIZE = 100

VEC_SIZE = KMER_SIZE if KMER_SIZE else SEQ_SIZE 
INPUT_SIZE = SEQ_SIZE
LATENT_DIMS = 2
EPOCHS = 600




# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# group_subs_rate= 0.1 * 1
# seqs, group1, group2 = generate_two_groups(SEQ_SIZE, group_subs_rate=group_subs_rate, subs_rate=0.1, kmer_k_size=KMER_SIZE)

# # TODO data
# data, _ = build_dataloader(seqs, KMER_SIZE)


# model = CNNAutoencoder(INPUT_SIZE, LATENT_DIMS).to(DEVICE)
# model = train(model, data, DEVICE)
