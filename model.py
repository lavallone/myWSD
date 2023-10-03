import torch
from torch import optim
from torch import nn
import pytorch_lightning as pl
from transformers import BertModel
from torchmetrics import Accuracy
import json
import wandb
import random

class WSD_Model(pl.LightningModule):
    def __init__(self, hparams):
        super(WSD_Model, self).__init__()
        self.save_hyperparameters(hparams)
        print(f"___________ {self.hparams.coarse_or_fine} MODEL ___________\n")
        self.encoder = BertModel.from_pretrained("bert-large-cased") # there's the possibility to not download it each time!
        
        # we set all parameters to be not trainable
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # classification HEAD
        self.batch_norm = nn.BatchNorm1d(4096)
        self.hidden_MLP = nn.Linear(4096, self.hparams.hidden_dim, bias=True)
        self.act_fun = nn.SiLU(inplace=True) # Swish activation function
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.num_senses = 106553 if self.hparams.coarse_or_fine == "coarse" else 117659 # sense inventory
        self.classifier = nn.Linear(self.hparams.hidden_dim, self.num_senses, bias=False) # final linear projection with no bias
        
        # mapping from 'id' to name of the 'sense'
        self.id2sense = json.load(open("data/mapping/cluster_id2sense.json", "r")) if self.hparams.coarse_or_fine == "coarse" else json.load(open("data/mapping/fine_id2sense.json", "r"))
        
        # validation ACCURACY
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_senses)
    
    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
               
    def forward(self, batch):
        text = batch["inputs"]
        embed_text = self.encoder(text["input_ids"].to(self.device), attention_mask=text["attention_mask"].to(self.device), token_type_ids=text["token_type_ids"].to(self.device), output_hidden_states=True)
        # I take the hidden representation of the last four layers of each token
        #embed_text = torch.stack(embed_text.hidden_states[-4:], dim=0).sum(dim=0) # sum
        #embed_text = torch.stack(embed_text.hidden_states[-4:], dim=0).sum(dim=0) # mean
        embed_text = torch.cat(embed_text.hidden_states[-4:], dim=-1) # concatenation --> (batch, seq_len, 1024*4=4096)
        
        encoder_output = embed_text.view(-1, embed_text.shape[-1]) # flattened embeddings --> (batch*seq_len, 4096)
        encoder_output_norm = self.batch_norm(encoder_output)
        hidden_output = self.dropout(self.act_fun(self.hidden_MLP(encoder_output_norm)))
        
        return self.classifier(hidden_output) # (batch*seq_len, num_senses)

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=self.hparams.lr)
        return {
            "optimizer": optimizer,
        }
    
    def training_step(self, batch, batch_idx):
        labels = batch["cluster_gold"] if self.hparams.coarse_or_fine == "coarse" else batch["fine_gold"]
        labels = torch.tensor(labels).view(-1).to(self.device)
        outputs = self(batch)
        cross_entropy_loss = nn.CrossEntropyLoss()
        loss = cross_entropy_loss(outputs, labels)
        self.log_dict({"loss" : loss})
        # since we only monitor the loss for the training phase, we don't need to write additional 
        # code in the 'training_epoch_end' function!
        return {'loss': loss}

    def predict(self, batch, filter_ids, labels):
        
        with torch.no_grad():
            outputs = self(batch)
            print(outputs.shape)
            # we first need to filter-out the outputs relative to the labels != -100 (the one we are interested in)
            print(labels.shape)
            mask = labels!=-100
            labels_of_interest = labels[mask]
            print(labels_of_interest.shape)
            filter_outputs = torch.index_select(outputs, 0, torch.tensor(labels_of_interest).to(self.device))
            print(filter_outputs.shape)
            print(filter_ids)
            filter_ids = [l for item in filter_ids for l in item]
            print(filter_ids)
            assert len(filter_ids) == len(filter_outputs)
            ris = []
            for i in range(len(filter_outputs)):
                # predict only over the possible homonym clusters...
                candidates_pred = torch.index_select(filter_outputs[i], 0, torch.tensor(filter_ids[i]).to(self.device))
                best_prediction = torch.argmax(candidates_pred, dim=0)
                ris.append(filter_ids[i][best_prediction.item()])
            return ris, labels_of_interest # list of predicted senses (expressed in indices)
    
    def validation_step(self, batch, batch_idx):
        labels = batch["cluster_gold"] if self.hparams.coarse_or_fine == "coarse" else batch["fine_gold"]
        labels = torch.tensor(labels).view(-1)
        outputs = self(batch)
        # LOSS
        val_loss = nn.CrossEntropyLoss(outputs, labels)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        # ACCURACY
        filter_ids = batch["cluster_candidates"] if self.hparams.coarse_or_fine == "coarse" else batch["fine_candidates"]
        preds, labels_of_interest = self.predict(batch, filter_ids, labels)
        
        # DEBUG info for one random item of the batch
        # item_idx = random.randint(0, len(labels)-1)
        # candidates = [self.id2sense[str(e)] for e in filter_ids[item_idx]]
        # gold = [self.id2sense[str(e)] for e in labels[item_idx]]
        # pred = [self.id2sense[str(preds[item_idx])]]
        # debug_infos = {"debug_infos" : str(candidates) + " | " + str(gold) + " | " + str(pred)}
        
        assert len(preds) == labels_of_interest.shape[0]
        self.accuracy.update(torch.tensor(preds), labels_of_interest)
        self.log("val_accuracy", self.accuracy, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        
        #return debug_infos
    
    # at the end of the epoch, we log the DEBUG infos of each batch
    # def validation_epoch_end(self, outputs):
    #     c = [str(i+1) for i in range(len(outputs))]
    #     data = []
    #     for i in range(len(outputs)):
    #         data.append(outputs[i]["debug_infos"])
    #     self.logger.experiment.log({f"debug_infos_{self.current_epoch}": wandb.Table(columns=c, data=[data])})