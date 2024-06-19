"""
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/variational_autoencoder/main.py
https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import plot_latent2
from datasets import build_dataloaders

torch.manual_seed(0)


class Encoder(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)

        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)


    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))

        mu = self.linear2(x)
        log_var = self.linear3(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, input_size)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.decoder = Decoder(input_size, latent_dims)
        self.encoder = Encoder(input_size, latent_dims)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)

        return x_hat, mu, log_var

    def loss_fn(self, x_hat, x, mu, log_var):
        x = x.view(x.size(0), -1) 
        MSE = F.binary_cross_entropy(x_hat, x, size_average=False)
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + KLD

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std


SEQ_SIZE = 150
KMER = 148

VEC_SIZE = KMER if KMER else SEQ_SIZE 
INPUT_SIZE = 4 * VEC_SIZE
LATENT_DIMS = 2
EPOCHS = 140

def train(model, data, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        for x, y in data:
            x = x.to(DEVICE)

            opt.zero_grad()

            # mu, log_var = model.encoder(x)
            # z = model.reparameterize(mu, log_var)
            # x_hat = model.decoder(z)
            x_hat, mu, log_var = model(x)
            loss = model.loss_fn(x_hat, x, mu, log_var)

            loss.backward()
            opt.step()
    return model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
data, _ = build_dataloaders(seq_size=SEQ_SIZE, group_subs_rate=0.7, subs_rate=0.1, kmer=KMER)

model = VariationalAutoencoder(INPUT_SIZE, LATENT_DIMS).to(DEVICE)
model = train(model, data)

plot_latent2(model, data, DEVICE)