from dataclasses import dataclass

@dataclass
class Hparams:
    
    ## dataloader params
    coarse_or_fine: str = "coarse" # coarse-grained or fine-grained task
    data_train: str = "data/wsd_datasets/training_sets/mapped_semcor.json" # train dataset path
    data_val: str = "data/wsd_datasets/evaluation_sets/mapped_semeval2007.json" # validation dataset path
    data_test: str = "data/wsd_datasets/evaluation_sets/mapped_ALLamended.json" # test dataset path
    batch_size: int = 32 # size of the batches
    n_cpu: int = 8 # number of cpu threads to use for the dataloaders
    pin_memory: bool = False # parameter to pin memory in dataloader
    
    ## train params
    lr: float = 1e-4
    precision: int = 16 # 16 or 32 precision training
    
    ## model params
    hidden_dim: int = 512 
    dropout: float = 0.6 # dropout value ?