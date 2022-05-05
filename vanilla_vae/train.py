from vanilla_vae.src.vae import VariationalAutoEncoder
from src import SimpleCNNEncoder, SimpleCNNDecoder, MNISTDataModule, LitProgressBar
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


latent_dims = 10

cnn_encoder = SimpleCNNEncoder(latent_dims=latent_dims)
cnn_decoder = SimpleCNNDecoder(latent_dims=latent_dims)

model = VariationalAutoEncoder(encoder=cnn_encoder, decoder=cnn_decoder)
data_module = MNISTDataModule()


trainer = Trainer(
    max_epochs=50,
    callbacks=[
        ModelCheckpoint(dirpath='artifacts/model', filename='vae_checkpoint',
                        monitor='val_loss', mode='min', save_top_k=1),
        LitProgressBar()
    ]
)
trainer.fit(model, data_module)
