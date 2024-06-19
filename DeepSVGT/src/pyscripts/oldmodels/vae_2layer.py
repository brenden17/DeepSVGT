"""
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/variational_autoencoder/main.py
"""
print('===')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions

from utils import plot_latent
from datasets import build_dataloaders

torch.manual_seed(0)


class Encoder(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)

        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        # self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        # self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))

        # return self.linear2(x), self.linear3(x)
        z_loc = self.linear2(x)
        z_scale = self.linear3(x)
        return z_loc, z_scale
        # mu =  self.linear2(x)
        # sigma = torch.exp(self.linear3(x))



        # z = mu + sigma*self.N.sample(mu.shape)

        # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        # return z


class Decoder(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, input_size)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z
        # return z.reshape((-1, 1, 100, 1200))


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.decoder = Decoder(input_size, latent_dims)
        self.encoder = Encoder(input_size, latent_dims)

    def forward(self, x):
        # z = self.encoder(x)
        # return self.decoder(z)

        z_loc, z_scale = self.encoder(x)
        z = self.reparameterize(z_loc, z_scale)
        x_reconst = self.decoder(z)

        return x_reconst, mu, log_var

    def loss_fn(self, x_hat, x, z_loc, z_scale):
        # Compute reconstruction loss and kl divergence
        # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
        # reconst_loss = F.binary_cross_entropy(x_hat, x, size_average=False)
        # kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # # Backprop and optimize
        # return reconst_loss + kl_div
        # MSE = F.mse_loss(x_hat, x, size_average=False)*10
        MSE = F.binary_cross_entropy(x_hat, x, size_average=False)
        KLD = -0.5 * torch.mean(1 + z_scale - z_loc.pow(2) - z_scale.exp())
        return MSE + KLD

    # def reparameterize(self, mu, log_var):
    #     std = torch.exp(log_var/2)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std

    def reparameterize(self, z_loc, z_scale):
        std = z_scale.mul(0.5).exp_()
        epsilon = torch.randn(*z_loc.size()).to(device)
        z = z_loc + std * epsilon
        return z

INPUT_SIZE = 1200
LATENT_DIMS = 2
EPOCHS = 20


def train(vae, data, epochs=EPOCHS):
    opt = torch.optim.Adam(vae.parameters())
    for epoch in range(epochs):
        for x, y in data:
            x = x.to(DEVICE).
            # x = x.to(DEVICE).view(-1, 1200)
            # x_hat = vae(x)
            # loss = ((x - x_hat)**2).sum() + vae.encoder.kl

            # x_reconst, mu, log_var = vae(x)

            z_loc, z_scale = vae.encoder(x)
            z = vae.reparameterize(z_loc, z_scale)
            x_hat = vae.decoder(z)
            loss = loss_fn(x_hat, x, z_loc, z_scale)

        
            # Compute reconstruction loss and kl divergence
            # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
            # reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
            # kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # # Backprop and optimize
            # loss = reconst_loss + kl_div
            vae.loss_fn()

            opt.zero_grad()

            loss.backward()
            opt.step()
    return vae

print('ss')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
data, _ = build_dataloaders()

vae = VariationalAutoencoder(INPUT_SIZE, LATENT_DIMS).to(DEVICE)
vae = train(vae, data)

plot_latent(vae, data, DEVICE)


print("DONE")