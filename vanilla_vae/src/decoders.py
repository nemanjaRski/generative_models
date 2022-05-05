import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule


class SimpleCNNDecoder(LightningModule):
    def __init__(self, latent_dims, linear_size=128):
        super(SimpleCNNDecoder, self).__init__()

        self.decoder_pipeline = nn.Sequential(
            nn.Linear(in_features=latent_dims, out_features=linear_size),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=linear_size, out_features=512),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(32, 4, 4)),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, output_padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=2, output_padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=5),
        )

    def forward(self, z):
        x_hat = self.decoder_pipeline(z)
        x_hat = torch.sigmoid(x_hat)
        return x_hat
