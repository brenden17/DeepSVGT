"""
https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import build_dataloader, kmer, BASE_SIZE
from utils import plot_latent, create_gif_plots, generate_latent, plot_sim_latent
from simulation import generate_two_groups



torch.manual_seed(0)


# class Encoder(nn.Module):
#     def __init__(self, input_size, latent_dims):
#         super(Encoder, self).__init__()
        
#         self.input_size = input_size

#         self.kernel_1 = 2
#         self.stride = 1
#         self.out_size = 20
#         # self.conv1 = nn.Conv1d(1, 16, 8, 2, padding=3)
#         self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)

#         # self.conv2 = nn.Conv1d(16, 16, 8, 2, padding=3)
#         # self.conv3 = nn.Conv1d(16, 32, 8, 2, padding=3)
#         # self.conv4 = nn.Conv1d(32, 32, 8, 2, padding=3)

#         out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
#         out_conv_1 = math.floor(out_conv_1)
#         out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
#         out_pool_1 = math.floor(out_pool_1)
#         print('out_pool_1-------------------{out_pool_1}')

#         self.fc1 = nn.Linear(out_pool_1, 64)
#         # self.fc1 = nn.Linear(240000, 64)
#         self.fc2 = nn.Linear(64, 16) 
#         self.fc21 = nn.Linear(16, latent_dims)
#         self.fc22 = nn.Linear(16, latent_dims)
#         self.relu = nn.ReLU()

#     # def out_size(self):
#     #     out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
#     #     out_conv_1 = math.floor(out_conv_1)
#     #     out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
#     #     out_pool_1 = math.floor(out_pool_1)

#     def forward(self, x):
#         # x = x.view(-1,1,319200)
#         print('1...')
#         print(x.shape)
#         # x = x.view(-1, 1, self.input_size)
#         # x = torch.flatten(x, start_dim=1)
#         # print(x.shape)
#         x = self.relu(self.conv1(x))
#         print('2...')
#         print(x.shape)
        
#         # x = self.relu(self.conv2(x))
#         # print('3...')
#         # print(x.shape)
        
#         # x = self.relu(self.conv3(x))
#         # print('4...')
#         # print(x.shape)
        
#         # x = self.relu(self.conv4(x))
#         # print("5...")
#         # print(x.shape)

#         # x = x.view(-1, 1, 240000)
#         s_shape = x.shape[0]
#         sss = 32 * 192 * s_shape

#         # x = x.view(-1, 1, 240000)
#         # x = x.view(-1, 32*19200)
#         s_shape = x.shape[0]
#         sss = 32 * 192 * s_shape
#         print(sss)
#         x = x.view(-1, sss)
        
#         print("6...")
#         print(x.shape)

#         x = self.relu(self.fc1(x))
#         x = F.dropout(x, 0.3)
#         x = self.relu(self.fc2(x))

#         z_loc = self.fc21(x)
#         z_scale = self.fc22(x)

#         print("6...z_loc")
#         print(z_loc.shape)
#         print("6...z_scale")
#         print(z_scale.shape)
#         return z_loc, z_scale

# class Decoder(nn.Module):
#     def __init__(self, latent_dims):
#         super(Decoder, self).__init__()
#         self.fc1 = nn.Linear(latent_dims, 32*19200)
#         # self.conv1 = nn.ConvTranspose1d(32, 32, 8, 2, padding=3)
#         self.conv1 = nn.ConvTranspose1d(32, 32, 8, 2, padding=3)
#         self.conv2 = nn.ConvTranspose1d(32, 16, 8, 2, padding=3)
#         self.conv3 = nn.ConvTranspose1d(16, 16, 8, 2, padding=3)
#         self.conv4 = nn.ConvTranspose1d(16, 1, 8, 2, padding=3)
#         self.relu = nn.ReLU()

#     def forward(self, z):

#         print("7...")
#         print(z.shape)
#         z = self.relu(self.fc1(z))
#         # z = z.view(-1, 240000)
#         print("8...")
#         print(z.shape)
#         # z = z.view(-1, 32, 75)
#         z = z.view(-1, 32, 19200)

#         print("9...")
#         print(z.shape)
        
#         z = self.relu(self.conv1(z))
#         print("10...")
#         print(z.shape)
        
#         z = self.relu(self.conv2(z))
#         print("11...")
#         print(z.shape)
#         z = self.relu(self.conv3(z))
#         print("12...")
#         print(z.shape)
        
#         z = self.conv4(z)
#         print("13...")
#         print(z.shape)
        
#         # z = self.conv5(z)
#         # print("14...")
#         # print(z.shape)
        
#         recon = torch.sigmoid(z)
#         return recon



# class CNNVariationalAutoencoder(nn.Module):
#     def __init__(self, input_size, latent_dims):
#         super(CNNVariationalAutoencoder, self).__init__()
#         self.input_size = input_size
#         self.decoder = Decoder(latent_dims)
#         self.encoder = Encoder(self.input_size, latent_dims)

#     def forward(self, x):
#         mu, log_var = self.encoder(x)
#         z = self.reparameterize(mu, log_var)
#         x_hat = self.decoder(z)

#         return x_hat, mu, log_var

#     def loss_fn(self, x_hat, x, mu, log_var):
#         # x = x.view(x.size(0), -1) 
#         x = x.view(-1,self.input_size)
#         x_hat = x_hat.view(-1,self.input_size)
#         # x = torch.flatten(x, start_dim=1)
#         print('x_hat')
#         print(x_hat.shape)
#         print('x')
#         print(x.shape)
        
#         MSE = F.binary_cross_entropy(x_hat, x, size_average=False)
#         # MSE = F.mse_loss(x_hat, x, size_average=False)*10
#         KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
#         return MSE + KLD

#     def reparameterize(self, mu, log_var): 
#         std = torch.exp(log_var/2)
#         eps = torch.randn_like(std)
#         return mu + eps * std


# # SEQ_SIZE = 96
# # KMER = 94

# # # VEC_SIZE = KMER if KMER else SEQ_SIZE 
# # INPUT_SIZE = BASE_SIZE * SEQ_SIZE
# # LATENT_DIMS = 2
# # EPOCHS = 440

# # GROUP_SUBS_RATE = 0.8
# # SUBS_RATE = 0.3

# # NOTE='CNN padding'


# def train(model, data, device, epochs=440):
#     opt = torch.optim.Adam(model.parameters())
#     for epoch in range(epochs):
#         for x, y in data:
#             x = x.to(device)

#             opt.zero_grad()

#             x_hat, mu, log_var = model(x)
#             loss = model.loss_fn(x_hat, x, mu, log_var)

#             loss.backward()
#             opt.step()
#     return model

# # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# # data, _ = build_dataloaders(seq_size=SEQ_SIZE, group_subs_rate=GROUP_SUBS_RATE, subs_rate=SUBS_RATE, kmer=KMER)

# # model = CNNVariationalAutoencoder(INPUT_SIZE, LATENT_DIMS).to(DEVICE)
# # model = train(model, data)


# # # plot_latent2(model, data, DEVICE, title)
# # x, y = generate_latent(model, data, DEVICE)




# # r = metrics_cluster(x, y)

# # r = ', '.join([f'group {item[0]}:{item[1]:.2f}' for item in r])
# # title = f'seq:{SEQ_SIZE}, kmer:{KMER}, epochs:{EPOCHS}, group:{GROUP_SUBS_RATE}, subs:{SUBS_RATE}, \nnote:{NOTE}, {r}'

# # plot_latent3(x, y, title)


def calcu_features(input_size, kernel_size, stride):
    '''Calculates the number of output features after Convolution + Max pooling
    
    Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
    Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
    
    source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    '''
    # Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
    out_conv = ((input_size - 1 * (kernel_size - 1) - 1) / stride) + 1
    out_conv = math.floor(out_conv)

    return out_conv
    # out_pool = ((out_conv_1 - 1 * (kernel_size - 1) - 1) / stride) + 1
    # out_pool = math.floor(out_pool)
    
    # return out_pool


class Encoder2Layers(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(Encoder2Layers, self).__init__()

        self.input_size = 6
        self.out_size = 10
        self.kernel_1 = 2
        self.stride = 1
        # self.conv1 = nn.Conv1d(1, 16, 8, 2, padding=3)

        self.conv_1 = nn.Conv1d(self.input_size, self.out_size, self.kernel_1, self.stride)

        out_pool_1 = calcu_features(self.input_size, self.kernel_1, self.stride)
        print(f'out_pool_1-------------------{out_pool_1}')

        # self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)


        self.linear1 = nn.Linear(990, 512)

        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)


    def forward(self, x):
        # x = torch.flatten(x, start_dim=1)
        print('befone conv 1')
        print(x.shape)
        batch, seq_size, bases = x.shape
        x = x.reshape(batch, bases, seq_size)
        x = self.conv_1(x)
        x = torch.relu(x)
        # x = self.pool_1(x)

        print('after conv 1...')
        print(x.shape)

        # x = x.view(-1,1,319200)
        x = torch.flatten(x, start_dim=1)
        print('222222')
        print(x.shape)


        x = F.relu(self.linear1(x))


        mu = self.linear2(x)
        log_var = self.linear3(x)
        return mu, log_var

class Decoder2Layers(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(Decoder2Layers, self).__init__()
        self.conv1 = nn.ConvTranspose1d(10, 6, 2, 1)

        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 990)

    def forward(self, z):
        print("22222...")
        print(z.shape)
        
        z = F.relu(self.linear1(z))

        print("8...")
        print(z.shape)

        x = torch.sigmoid(self.linear2(z))

        print("9...")
        print(x.shape)

        x = x.view(100, 10, 99)

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
        self.encoder = Encoder2Layers(input_size, latent_dims)
        self.decoder = Decoder2Layers(input_size, latent_dims)

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
INPUT_SIZE = BASE_SIZE * VEC_SIZE
LATENT_DIMS = 2
EPOCHS = 600




DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

group_subs_rate= 0.1 * 1
seqs, group1, group2 = generate_two_groups(SEQ_SIZE, group_subs_rate=group_subs_rate, subs_rate=0.1, kmer_k_size=KMER_SIZE)

# TODO data
data, _ = build_dataloader(seqs, KMER_SIZE)


model = CNNAutoencoder(INPUT_SIZE, LATENT_DIMS).to(DEVICE)
model = train(model, data, DEVICE)
