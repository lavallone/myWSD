import wandb
from argparse import ArgumentParser
from data_module import WSD_DataModule
from hyperparameters import Hparams
from train import train_model
from model import WSD_Model
from evaluation import base_evaluation, fine2cluster_evaluation, cluster_filter_evaluation

import torch
import random
import numpy as np
import wandb
from dataclasses import asdict
import pytorch_lightning as pl

# setting the seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    _ = pl.seed_everything(seed)

def parse_args():
    parser = ArgumentParser()
    # experiments parameters
    parser.add_argument("--mode", type=str, default="train", help="train/eval", required=True)
    parser.add_argument("--type", type=str, default="coarse", help="coarse/fine/fine2cluster/cluster_filter", required=True)
    parser.add_argument("--encoder", type=str, default="bert", help="encoder type", required=True)
    parser.add_argument("--model", type=str, default="checkpoints/coarse.ckpt", help="path to model", required=False)
    parser.add_argument("--model2", type=str, default="checkpoints/coarse.ckpt", help="path to second model", required=False)
    parser.add_argument("--oracle", type=str, default="False", help="use or not the oracle in the 'cluster_filter' scenario", required=False)
    return parser.parse_args()


def main(arguments):
    if arguments.mode == "train":
        wandb.login() # this is the key to paste each time for login: 65a23b5182ca8ce3eb72530af592cf3bfa19de85

        version_name = arguments.type+"_"+arguments.encoder
        with wandb.init(entity="lavallone", project="homonyms", name=version_name, mode="online"):
            hparams = asdict(Hparams())
            hparams["encoder"] = arguments.encoder
            data = WSD_DataModule(hparams)
            model = WSD_Model(hparams)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            train_model(data, model, experiment_name=version_name, metric_to_monitor="val_accuracy", mode="max", epochs=50, precision=hparams["precision"])

        wandb.finish()
    
    elif arguments.mode == "eval":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if arguments.type == "coarse" or arguments.type == "fine":
            ckpt = arguments.model
            model = WSD_Model.load_from_checkpoint(ckpt).to(device)
            data = WSD_DataModule(model.hparams)
            data.setup()
            base_evaluation(model, data)
        
        elif arguments.type == "fine2cluster":
            ckpt = arguments.model
            model = WSD_Model.load_from_checkpoint(ckpt).to(device)
            assert model.hparams.coarse_or_fine == "fine"
            data = WSD_DataModule(model.hparams)
            data.setup()
            # evaluation on homonym clusters using a fine-grained model
            fine2cluster_evaluation(model, data)
            
        elif arguments.type == "cluster_filter":
            ckpt1 = arguments.model
            ckpt2 = arguments.model2
            coarse_model = WSD_Model.load_from_checkpoint(ckpt1).to(device)
            assert coarse_model.hparams.coarse_or_fine == "coarse"
            fine_model = WSD_Model.load_from_checkpoint(ckpt2).to(device)
            assert fine_model.hparams.coarse_or_fine == "fine"
            data = WSD_DataModule(coarse_model.hparams)
            data.setup()
            # evaluation on fine senses using a coarse model for filtering out
            cluster_filter_evaluation(coarse_model, fine_model, data, oracle_or_not=bool(arguments.oracle))

if __name__ == '__main__':
    set_seed(99)
    arguments = parse_args()
    main(arguments)