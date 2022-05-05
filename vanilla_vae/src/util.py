from pytorch_lightning.callbacks import ProgressBar
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch


class LitProgressBar(ProgressBar):
    """
     progress bar fixing hack from
     https://stackoverflow.com/questions/59455268/how-to-disable-progress-bar-in-pytorch-lightning/66731318#66731318
    """
    def init_validation_tqdm(self):
        bar = tqdm(
            disable=True,
        )
        return bar


def plot_vae_outputs(x, x_hat, idx, dir_name, img_name, n=5):
    plt.figure(figsize=(10, 4.5))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = x[i].to('cpu')
        rec_img = x_hat[i].to('cpu')

        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Original images')
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Reconstructed images')
    plt.savefig(f'{dir_name}/{img_name}-{idx}.jpg')
    plt.close('all')


def plot_vae_samples(decoder, latent_dims, idx, dir_name, img_name, n=5):
    plt.figure(figsize=(10, 1.5))
    distribution = torch.distributions.Normal(0, 1)
    for i in range(n):
        rec_img = decoder(distribution.sample((1, latent_dims)))
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(rec_img.cpu().squeeze().detach().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(f'{dir_name}/{img_name}-{idx}.jpg', bbox_inches='tight')
    plt.close()


def mse_loss(x, x_hat, mu, sigma, kl_weight, reconstruction_weight):

    reconstruction_loss = torch.nn.MSELoss(reduction='none')(x_hat, x)
    kl_divergence = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()

    loss = kl_weight * kl_divergence + reconstruction_weight * reconstruction_loss

    return loss


def bce_loss(x, x_hat, mu, sigma, kl_weight, reconstruction_weight):

    reconstruction_loss = torch.nn.BCELoss()(x_hat, x)
    kl_divergence = -0.5 * torch.mean(1.0 + sigma - mu.pow(2.0) - sigma.exp())

    loss = kl_weight * kl_divergence + reconstruction_weight * reconstruction_loss

    return loss
