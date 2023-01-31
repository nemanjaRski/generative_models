from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self, channel_sizes, *args, **kwargs):
        super(Encoder, self).__init__()
        self.encoder_pipeline = nn.Sequential(
            *[self.conv_block(in_channels, out_channels, *args, **kwargs)
              for in_channels, out_channels in zip(channel_sizes, channel_sizes[1:])])

    def forward(self, x):
        return self.encoder_pipeline(x)

    @staticmethod
    def conv_block(in_channels, out_channels, activation='relu', *args, **kwargs):
        activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['relu', nn.ReLU()]
        ])

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            activations[activation]
        )


class Decoder(nn.Module):
    def __init__(self, channel_sizes, *args, **kwargs):
        super(Decoder, self).__init__()

        self.pipeline = nn.Sequential(
            *[self.conv_transpose_block(in_channels, out_channels, *args, **kwargs)
              for in_channels, out_channels in zip(channel_sizes, channel_sizes[1:])])

    def forward(self, x):
        return self.pipeline(x)

    @staticmethod
    def conv_transpose_block(in_channels, out_channels, activation='relu', *args, **kwargs):
        activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['relu', nn.ReLU()]
        ])

        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            activations[activation]
        )


class OutputLayer(nn.Module):
    def __init__(self, transform_channels, output_channels):
        super(OutputLayer, self).__init__()

        self.pipeline = nn.Sequential(
            nn.ConvTranspose2d(transform_channels,
                               transform_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(transform_channels),
            nn.LeakyReLU(),
            nn.Conv2d(transform_channels, out_channels=output_channels,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def forward(self, x):
        return self.pipeline(x)


class VariationalAutoEncoder(nn.Module):

    def __init__(self, latent_dim: int, channel_sizes: List[int]) -> None:

        super(VariationalAutoEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.channel_sizes = channel_sizes

        self.encoder = Encoder(self.channel_sizes,
                               kernel_size=3,
                               padding=1,
                               stride=2,
                               activation='lrelu')

        self.fc_mu = nn.Linear(self.channel_sizes[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(self.channel_sizes[-1] * 4, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.channel_sizes[-1] * 4)

        self.channel_sizes.reverse()

        self.decoder = Decoder(channel_sizes=self.channel_sizes[:-1],
                               kernel_size=3,
                               stride=2,
                               output_padding=1,
                               padding=1,
                               activation='lrelu')

        self.output_layer = OutputLayer(self.channel_sizes[-2],
                                        self.channel_sizes[-1])

    def forward(self, x: Tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def encode(self, x: Tensor):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        z = self.decoder_input(z)
        z = z.unflatten(1, (512, 2, 2))
        z = self.decoder(z)
        x_hat = self.output_layer(z)
        return x_hat

    @staticmethod
    def reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, num_samples: int, current_device: int) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    @staticmethod
    def loss_function(x_hat, x, mu, log_var, recon_weight, kl_weight) -> tuple:
        recon_loss = F.mse_loss(x_hat, x)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recon_weight * recon_loss + kl_weight * kl_loss
        return loss, recon_loss.detach(), -kl_loss.detach()



