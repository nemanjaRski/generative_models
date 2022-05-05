import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule


class SimpleCNNEncoder(LightningModule):
    def __init__(self, latent_dims, linear_size=128):
        super(SimpleCNNEncoder, self).__init__()

        self.encoder_pipeline = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=512, out_features=linear_size),
            nn.ReLU(inplace=True)
        )

        self.N = torch.distributions.Normal(0, 1)

        self.linear_mu = nn.Linear(in_features=linear_size, out_features=latent_dims)
        self.linear_sigma = nn.Linear(in_features=linear_size, out_features=latent_dims)

    def forward(self, x):
        x = self.encoder_pipeline(x)
        mu = self.linear_mu(x)
        sigma = torch.exp(self.linear_sigma(x))
        z = mu + sigma * self.N.sample(mu.shape)
        return z, mu, sigma
