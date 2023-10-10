from dataclasses import dataclass

@dataclass
class Hparams:
    
    ## dataloader params
    coarse_or_fine: str = "fine" # coarse-grained or fine-grained task
    data_train: str = "data_v3/wsd_datasets/train.json" # train dataset path
    data_val: str = "data_v3/wsd_datasets/dev.json" # validation dataset path
    data_test: str = "data_v3/wsd_datasets/test.json" # test dataset path
    batch_size: int = 8 # size of the batches
    n_cpu: int = 8 # number of cpu threads to use for the dataloaders
    pin_memory: bool = False # parameter to pin memory in dataloader
    
    ## train params
    lr: float = 1e-4
    precision: int = 32 # 16 or 32 precision training
    
    ## model params
    encoder: str = "deberta" # bert, roberta, deberta, electra
    hidden_dim: int = 512 
    dropout: float = 0.1 # dropout value