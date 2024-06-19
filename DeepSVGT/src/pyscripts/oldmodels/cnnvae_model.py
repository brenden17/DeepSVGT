import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions

torch.manual_seed(0)

"""
A Convolutional Variational Autoencoder
# """
# class VAE(nn.Module):
    # def __init__(self, imgChannels=1, featureDim=32*20*20, zDim=256):
    #     super(VAE, self).__init__()

    #     # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
    #     self.encConv1 = nn.Conv1d(imgChannels, 16, 5)
    #     self.encConv2 = nn.Conv1d(16, 32, 5)
    #     self.encFC1 = nn.Linear(featureDim, zDim)
    #     self.encFC2 = nn.Linear(featureDim, zDim)

    #     # Initializing the fully-connected layer and 2 convolutional layers for decoder
    #     self.decFC1 = nn.Linear(zDim, featureDim)
    #     self.decConv1 = nn.ConvTranspose1d(32, 16, 5)
    #     self.decConv2 = nn.ConvTranspose1d(16, imgChannels, 5)

    # def encoder(self, x):

    #     # Input is fed into 2 convolutional layers sequentially
    #     # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
    #     # Mu and logVar are used for generating middle representation z and KL divergence loss
    #     x = F.relu(self.encConv1(x))
    #     x = F.relu(self.encConv2(x))
    #     x = x.view(-1, 32*20*20)
    #     mu = self.encFC1(x)
    #     logVar = self.encFC2(x)
    #     return mu, logVar

    # def reparameterize(self, mu, logVar):

    #     #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
    #     std = torch.exp(logVar/2)
    #     eps = torch.randn_like(std)
    #     return mu + std * eps

    # def decoder(self, z):

    #     # z is fed back into a fully-connected layers and then into two transpose convolutional layers
    #     # The generated output is the same size of the original input
    #     x = F.relu(self.decFC1(z))
    #     x = x.view(-1, 32, 20, 20)
    #     x = F.relu(self.decConv1(x))
    #     x = torch.sigmoid(self.decConv2(x))
    #     return x

    # def forward(self, x):

    #     # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
    #     # output, mu, and logVar are returned for loss computation
    #     mu, logVar = self.encoder(x)
    #     z = self.reparameterize(mu, logVar)
    #     out = self.decoder(z)
    #     return out, mu, logVar



class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv1d(16, 16, 8, 2, padding=3)
        self.conv3 = nn.Conv1d(16, 32, 8, 2, padding=3)
        self.conv4 = nn.Conv1d(32, 32, 8, 2, padding=3)

        self.fc1 = nn.Linear(32*21, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc21 = nn.Linear(16, z_dim)
        self.fc22 = nn.Linear(16, z_dim)
        self.relu = nn.ReLU()


    def forward(self, x):
        # x = x.view(-1,1,336)
        x = x.view(-1,1,1200)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.view(-1, 672)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, 0.3)
        x = self.relu(self.fc2(x))
        z_loc = self.fc21(x)
        z_scale = self.fc22(x)
        return z_loc, z_scale

class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, 672)
        self.conv1 = nn.ConvTranspose1d(32, 32, 8, 2, padding=3)
        self.conv2 = nn.ConvTranspose1d(32, 32, 8, 2, padding=3)
        self.conv3 = nn.ConvTranspose1d(32, 16, 8, 2, padding=3)
        self.conv4 = nn.ConvTranspose1d(16, 16, 8, 2, padding=3)
        self.conv5 = nn.ConvTranspose1d(16, 1, 7, 1, padding=3)
        self.relu = nn.ReLU()

    def forward(self, z):
        z = self.relu(self.fc1(z))
        z = z.view(-1, 32, 21)
        z = self.relu(self.conv1(z))
        z = self.relu(self.conv2(z))
        z = self.relu(self.conv3(z))
        z = self.relu(self.conv4(z))
        z = self.conv5(z)
        recon = torch.sigmoid(z)
        return recon


class VAE(nn.Module):
    def __init__(self, z_dim=2):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.cuda()
        self.z_dim = z_dim

    def reparameterize(self, z_loc, z_scale):
        std = z_scale.mul(0.5).exp_()
        epsilon = torch.randn(*z_loc.size()).to(device)
        z = z_loc + std * epsilon
        return z
