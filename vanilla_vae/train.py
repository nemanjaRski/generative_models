import yaml
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src import CelebADataModule
from src.ligtning_wrapper import LightningWrapper
from src.vae import VariationalAutoEncoder


def train():
    with open('params.yaml', 'r') as file:
        try:
            params = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    artifacts_dir = params['logging_params']['save_dir']
    run_name = params['dataset']['name']

    tb_logger = TensorBoardLogger(save_dir=artifacts_dir,
                                  name=run_name)

    Path(f"{tb_logger.log_dir}/train_samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/train_reconstructions").mkdir(exist_ok=True, parents=True)

    model = VariationalAutoEncoder(params['train_params']['latent_dim'], params['train_params']['channel_sizes'])

    checkpoint_path = params['train_params']['checkpoint']
    if checkpoint_path:
        experiment = LightningWrapper.load_from_checkpoint(checkpoint_path)
    else:
        experiment = LightningWrapper(model, device, params['train_params'])

    data_module = CelebADataModule(**params['batch'])

    trainer = Trainer(
        logger=tb_logger,
        max_epochs=params["train_params"]['max_epochs'],
        callbacks=[
            ModelCheckpoint(dirpath=f'{tb_logger.log_dir}/model', filename='vae_checkpoint',
                            monitor='val_loss', mode='min', save_top_k=1, save_last=True),
        ],
        accelerator='gpu',
        devices=1
    )

    trainer.fit(experiment, data_module)


if __name__ == "__main__":
    train()
