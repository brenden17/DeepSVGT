"""
https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable

from datasets import build_dataloader, kmer, BASE_SIZE
from utils import plot_latent, create_gif_plots, generate_latent, plot_sim_latent
from simulation import generate_two_groups


torch.manual_seed(0)


class NearestEmbedFunc(Function):
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input, emb):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(
                emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *
                         list(input.shape[2:]), input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)
                                      ).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) ==
                           latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(
                ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None


def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)


class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):
        super(NearestEmbed, self).__init__()
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight)


# adapted from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py#L25
# that adapted from https://github.com/deepmind/sonnet


class NearestEmbedEMA(nn.Module):
    def __init__(self, n_emb, emb_dim, decay=0.99, eps=1e-5):
        super(NearestEmbedEMA, self).__init__()
        self.decay = decay
        self.eps = eps
        self.embeddings_dim = emb_dim
        self.n_emb = n_emb
        self.emb_dim = emb_dim
        embed = torch.rand(emb_dim, n_emb)
        self.register_buffer('weight', embed)
        self.register_buffer('cluster_size', torch.zeros(n_emb))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, x):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """

        dims = list(range(len(x.size())))
        x_expanded = x.unsqueeze(-1)
        num_arbitrary_dims = len(dims) - 2
        if num_arbitrary_dims:
            emb_expanded = self.weight.view(
                self.emb_dim, *([1] * num_arbitrary_dims), self.n_emb)
        else:
            emb_expanded = self.weight

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [x.shape[0], *list(x.shape[2:]), x.shape[1]]
        result = self.weight.t().index_select(
            0, argmin.view(-1)).view(shifted_shape).permute(0, dims[-1], *dims[1:-1])

        if self.training:
            latent_indices = torch.arange(self.n_emb).type_as(argmin)
            emb_onehot = (argmin.view(-1, 1) ==
                          latent_indices.view(1, -1)).type_as(x.data)
            n_idx_choice = emb_onehot.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            flatten = x.permute(
                1, 0, *dims[-2:]).contiguous().view(x.shape[1], -1)

            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, n_idx_choice
            )
            embed_sum = flatten @ emb_onehot
            self.embed_avg.data.mul_(self.decay).add_(
                1 - self.decay, embed_sum)

            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) /
                (n + self.n_emb * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.weight.data.copy_(embed_normalized)

        return result, 



class VQ_VAE(nn.Module):
    """Vector Quantized AutoEncoder for mnist"""

    def __init__(self, input_size, hidden=200, k=10, vq_coef=0.2, comit_coef=0.4, **kwargs):
        super(VQ_VAE, self).__init__()

        self.input_size = input_size
        self.emb_size = k
        self.fc1 = nn.Linear(self.input_size, 400)
        self.fc2 = nn.Linear(400, hidden)
        self.fc3 = nn.Linear(hidden, 400)
        self.fc4 = nn.Linear(400, self.input_size)

        self.emb = NearestEmbed(k, self.emb_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.vq_coef = vq_coef
        self.comit_coef = comit_coef
        self.hidden = hidden
        self.ce_loss = 0
        self.vq_loss = 0
        self.commit_loss = 0

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.fc2(h1)
        print(self.hidden, self.emb_size)
        return h2.view(-1, self.emb_size, int(self.hidden / self.emb_size))

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.tanh(self.fc4(h3))

    def forward(self, x):
        z_e = self.encode(x.view(-1, self.input_size))
        z_q, _ = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        return self.decode(z_q), z_e, emb

    def sample(self, size):
        sample = torch.randn(size, self.emb_size,
                             int(self.hidden / self.emb_size))
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        sample = self.decode(emb(sample).view(-1, self.hidden)).cpu()
        return sample

    def loss_function(self, x, recon_x, z_e, emb):
        self.ce_loss = F.binary_cross_entropy(recon_x, x.view(-1, self.input_size))
        self.vq_loss = F.mse_loss(emb, z_e.detach())
        self.commit_loss = F.mse_loss(z_e, emb.detach())

        return self.ce_loss + self.vq_coef*self.vq_loss + self.comit_coef*self.commit_loss

    def latest_losses(self):
        return {'cross_entropy': self.ce_loss, 'vq': self.vq_loss, 'commitment': self.commit_loss}



SEQ_SIZE = 100
KMER_SIZE = 100

VEC_SIZE = KMER_SIZE if KMER_SIZE else SEQ_SIZE 
INPUT_SIZE = BASE_SIZE * VEC_SIZE
LATENT_DIMS = 2
EPOCHS = 600



def train(model, data, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        for x, y in data:
            x = x.to(DEVICE)

            opt.zero_grad()

            x_hat, mu, log_var = model(x)
            loss = model.loss_fn(x_hat, x, mu, log_var)

            loss.backward()
            opt.step()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
    return model



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

group_subs_rate= 0.1 * 1
seqs, group1, group2 = generate_two_groups(SEQ_SIZE, group_subs_rate=group_subs_rate, subs_rate=0.1, kmer_k_size=KMER_SIZE)

# TODO data
data, _ = build_dataloader(seqs, KMER_SIZE)


model = VQ_VAE(INPUT_SIZE).to(DEVICE)
model = train(model, data)

# plot_latent2(model, data, DEVICE)