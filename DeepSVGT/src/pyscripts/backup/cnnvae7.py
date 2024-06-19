"""
https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import plot_latent2, plot_latent3, generate_latent, metrics_cluster
from datasets import build_dataloaders, BASE_SIZE


torch.manual_seed(0)


class Encoder(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv1d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv1d(16, 16, 8, 2, padding=3)
        self.conv3 = nn.Conv1d(16, 32, 8, 2, padding=3)
        self.conv4 = nn.Conv1d(32, 32, 8, 2, padding=3)

        self.fc1 = nn.Linear(32*19200, 64)
        # self.fc1 = nn.Linear(240000, 64)
        self.fc2 = nn.Linear(64, 16) 
        self.fc21 = nn.Linear(16, latent_dims)
        self.fc22 = nn.Linear(16, latent_dims)
        self.relu = nn.ReLU()


    def forward(self, x):
        # x = x.view(-1,1,319200)
        print('1...')
        print(x.shape)
        x = x.view(-1, 1, self.input_size)
        # x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        x = self.relu(self.conv1(x))
        print('2...')
        print(x.shape)
        
        x = self.relu(self.conv2(x))
        print('3...')
        print(x.shape)
        
        x = self.relu(self.conv3(x))
        print('4...')
        print(x.shape)
        
        x = self.relu(self.conv4(x))
        print("5...")
        print(x.shape)

        # x = x.view(-1, 1, 240000)
        s_shape = x.shape[0]
        sss = 32 * 192 * s_shape

        # x = x.view(-1, 1, 240000)
        # x = x.view(-1, 32*19200)
        s_shape = x.shape[0]
        sss = 32 * 192 * s_shape
        print(sss)
        x = x.view(-1, sss)
        
        print("6...")
        print(x.shape)

        x = self.relu(self.fc1(x))
        x = F.dropout(x, 0.3)
        x = self.relu(self.fc2(x))

        z_loc = self.fc21(x)
        z_scale = self.fc22(x)

        print("6...z_loc")
        print(z_loc.shape)
        print("6...z_scale")
        print(z_scale.shape)
        return z_loc, z_scale

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dims, 32*19200)
        # self.conv1 = nn.ConvTranspose1d(32, 32, 8, 2, padding=3)
        self.conv1 = nn.ConvTranspose1d(32, 32, 8, 2, padding=3)
        self.conv2 = nn.ConvTranspose1d(32, 16, 8, 2, padding=3)
        self.conv3 = nn.ConvTranspose1d(16, 16, 8, 2, padding=3)
        self.conv4 = nn.ConvTranspose1d(16, 1, 8, 2, padding=3)
        self.relu = nn.ReLU()

    def forward(self, z):

        print("7...")
        print(z.shape)
        z = self.relu(self.fc1(z))
        # z = z.view(-1, 240000)
        print("8...")
        print(z.shape)
        # z = z.view(-1, 32, 75)
        z = z.view(-1, 32, 19200)

        print("9...")
        print(z.shape)
        
        z = self.relu(self.conv1(z))
        print("10...")
        print(z.shape)
        
        z = self.relu(self.conv2(z))
        print("11...")
        print(z.shape)
        z = self.relu(self.conv3(z))
        print("12...")
        print(z.shape)
        
        z = self.conv4(z)
        print("13...")
        print(z.shape)
        
        # z = self.conv5(z)
        # print("14...")
        # print(z.shape)
        
        recon = torch.sigmoid(z)
        return recon



class CNNVariationalAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(CNNVariationalAutoencoder, self).__init__()
        self.input_size = input_size
        self.decoder = Decoder(latent_dims)
        self.encoder = Encoder(self.input_size, latent_dims)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)

        return x_hat, mu, log_var

    def loss_fn(self, x_hat, x, mu, log_var):
        # x = x.view(x.size(0), -1) 
        x = x.view(-1,self.input_size)
        x_hat = x_hat.view(-1,self.input_size)
        # x = torch.flatten(x, start_dim=1)
        print('x_hat')
        print(x_hat.shape)
        print('x')
        print(x.shape)
        
        MSE = F.binary_cross_entropy(x_hat, x, size_average=False)
        # MSE = F.mse_loss(x_hat, x, size_average=False)*10
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + KLD

    def reparameterize(self, mu, log_var): 
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std


# SEQ_SIZE = 96
# KMER = 94

# # VEC_SIZE = KMER if KMER else SEQ_SIZE 
# INPUT_SIZE = BASE_SIZE * SEQ_SIZE
# LATENT_DIMS = 2
# EPOCHS = 440

# GROUP_SUBS_RATE = 0.8
# SUBS_RATE = 0.3

# NOTE='CNN padding'


def train(model, data, device, epochs=440):
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        for x, y in data:
            x = x.to(device)

            opt.zero_grad()

            x_hat, mu, log_var = model(x)
            loss = model.loss_fn(x_hat, x, mu, log_var)

            loss.backward()
            opt.step()
    return model

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# data, _ = build_dataloaders(seq_size=SEQ_SIZE, group_subs_rate=GROUP_SUBS_RATE, subs_rate=SUBS_RATE, kmer=KMER)

# model = CNNVariationalAutoencoder(INPUT_SIZE, LATENT_DIMS).to(DEVICE)
# model = train(model, data)


# # plot_latent2(model, data, DEVICE, title)
# x, y = generate_latent(model, data, DEVICE)




# r = metrics_cluster(x, y)

# r = ', '.join([f'group {item[0]}:{item[1]:.2f}' for item in r])
# title = f'seq:{SEQ_SIZE}, kmer:{KMER}, epochs:{EPOCHS}, group:{GROUP_SUBS_RATE}, subs:{SUBS_RATE}, \nnote:{NOTE}, {r}'

# plot_latent3(x, y, title)