import os
import random
import torch
from torch import Tensor
import pytorch_lightning as pl
import torchvision.utils as vutils


class LightningWrapper(pl.LightningModule):

    def __init__(self,
                 vae_model,
                 device,
                 params: dict) -> None:
        super(LightningWrapper, self).__init__()

        self.model = vae_model
        self.params = params
        self.run_device = device
        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params['lr'])

    def training_step(self, batch, batch_idx):
        x, _ = batch

        x_hat, mu, log_var = self.forward(x)
        loss, recon_loss, kl_loss = self.model.loss_function(x_hat=x_hat,
                                                             x=x,
                                                             mu=mu,
                                                             log_var=log_var,
                                                             kl_weight=self.params['kl_weight'],
                                                             recon_weight=self.params['recon_weight'])

        self.log_dict({
            'train_loss': loss,
            'train_recon_loss': recon_loss,
            'train_kl_loss': kl_loss
        }, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        x_hat, mu, log_var = self.forward(x)
        loss, recon_loss, kl_loss = self.model.loss_function(x_hat=x_hat,
                                                             x=x,
                                                             mu=mu,
                                                             log_var=log_var,
                                                             kl_weight=1.0,
                                                             recon_weight=1.0)

        self.log_dict({
            'val_loss': loss,
            'val_recon_loss': recon_loss,
            'val_kl_loss': kl_loss
        }, sync_dist=True)

        return x_hat

    def log_samples(self, x_hat_sample):
        vutils.save_image(x_hat_sample,
                          os.path.join(self.logger.log_dir,
                                       'train_reconstructions',
                                       f'recons_{self.logger.name}_epoch_{self.current_epoch}.png'),
                          normalize=True,
                          nrow=8)

        samples = self.model.sample(150, x_hat_sample.get_device())
        vutils.save_image(samples.cpu().data,
                          os.path.join(self.logger.log_dir,
                                       'train_samples',
                                       f'sample_{self.logger.name}_epoch_{self.current_epoch}.png'),
                          normalize=True,
                          nrow=10)

    def validation_epoch_end(self, val_step_outputs):
        x_hat_list = []
        for batch in val_step_outputs:
            step_x_hat = batch
            x_hat_list.append(step_x_hat)

        self.log_samples(random.choice(x_hat_list))




