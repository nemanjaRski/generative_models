from vanilla_vae.src.vae import VariationalAutoEncoder
from src import SimpleCNNEncoder, SimpleCNNDecoder, plot_vae_samples
import torch


latent_dims = 10

cnn_encoder = SimpleCNNEncoder(latent_dims=latent_dims)
cnn_decoder = SimpleCNNDecoder(latent_dims=latent_dims)

model = VariationalAutoEncoder(encoder=cnn_encoder, decoder=cnn_decoder)
model.load_state_dict(torch.load('artifacts/model/vae_checkpoint.ckpt')['state_dict'])


plot_vae_samples(model.decoder, latent_dims, 0, 'artifacts/inference_examples/', 'img', 10)
