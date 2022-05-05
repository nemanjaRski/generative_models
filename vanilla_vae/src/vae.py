import pytorch_lightning as pl
import torch
from .util import plot_vae_outputs


def default_loss(x, x_hat, mu, sigma, kl_weight, reconstruction_weight):

    reconstruction_loss = ((x - x_hat) ** 2).sum()
    kl_divergence = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()

    loss = kl_weight * kl_divergence + reconstruction_weight * reconstruction_loss

    return loss


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder, loss_function=None):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if loss_function:
            self.loss_function = loss_function
        else:
            self.loss_function = default_loss

    def forward(self, x):
        z, mu, sigma = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mu, sigma

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

    def training_step(self, batch, batch_idx):
        x, _ = batch

        z, mu, sigma = self.encoder(x)
        x_hat = self.decoder(z)

        loss = self.loss_function(x, x_hat, mu, sigma, 1, 10)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        z, mu, sigma = self.encoder(x)
        x_hat = self.decoder(z)

        loss = self.loss_function(x, x_hat, mu, sigma, 1, 10)

        return loss, x, x_hat

    def validation_epoch_end(self, val_step_outputs):
        loss_list, x_list, x_hat_list = [], [], []
        for batch in val_step_outputs:
            step_loss, step_x, step_x_hat = batch
            loss_list.append(step_loss)
            x_list.append(step_x)
            x_hat_list.append(step_x_hat)

        plot_vae_outputs(x_list[0], x_hat_list[0], self.current_epoch, 'artifacts/train_examples', 'reconstruction')
        loss = torch.stack(loss_list).mean()
        self.log("val_loss", loss)
