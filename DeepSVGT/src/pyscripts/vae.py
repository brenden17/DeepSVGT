"""
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/variational_autoencoder/main.py
https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

class Encoder6Layers(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(Encoder6Layers, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 32)

        self.linear6 = nn.Linear(32, latent_dims)
        self.linear7 = nn.Linear(32, latent_dims)


    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))

        mu = self.linear6(x)
        log_var = self.linear7(x)
        return mu, log_var

class Decoder6Layers(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(Decoder6Layers, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, 256)
        self.linear5 = nn.Linear(256, 512)
        self.linear6 = nn.Linear(512, input_size)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))
        z = F.relu(self.linear4(z))
        z = F.relu(self.linear5(z))
        z = torch.sigmoid(self.linear6(z))
        return z

class Variational6LayersAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(Variational6LayersAutoencoder, self).__init__()
        self.encoder = Encoder6Layers(input_size, latent_dims)
        self.decoder = Decoder6Layers(input_size, latent_dims)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decoder(z)

        return x_reconst, mu, log_var

    def loss_fn(self, x_hat, x, mu, log_var):
        x = x.view(x.size(0), -1) 
        MSE = F.binary_cross_entropy(x_hat, x, size_average=False)
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + KLD

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std



class Encoder4Layers(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(Encoder4Layers, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)

        self.linear4 = nn.Linear(128, latent_dims)
        self.linear5 = nn.Linear(128, latent_dims)


    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        mu = self.linear4(x)
        log_var = self.linear5(x)
        return mu, log_var

class Decoder4Layers(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(Decoder4Layers, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 512)
        self.linear4 = nn.Linear(512, input_size)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))
        z = torch.sigmoid(self.linear4(z))
        return z

class Variational4LayersAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(Variational4LayersAutoencoder, self).__init__()
        self.encoder = Encoder4Layers(input_size, latent_dims)
        self.decoder = Decoder4Layers(input_size, latent_dims)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decoder(z)

        return x_reconst, mu, log_var

    def loss_fn(self, x_hat, x, mu, log_var):
        x = x.view(x.size(0), -1) 
        MSE = F.binary_cross_entropy(x_hat, x, size_average=False)
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + KLD

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std


class Encoder2Layers(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(Encoder2Layers, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)

        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)


    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))

        mu = self.linear2(x)
        log_var = self.linear3(x)
        return mu, log_var

class Decoder2Layers(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(Decoder2Layers, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, input_size)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z

class Variational2LayersAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(Variational2LayersAutoencoder, self).__init__()
        self.encoder = Encoder2Layers(input_size, latent_dims)
        self.decoder = Decoder2Layers(input_size, latent_dims)

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