import torch
from torch import optim
from torch import nn
import pytorch_lightning as pl
from transformers import BertModel
from torchmetrics import Accuracy
import json
import wandb
import random
from evaluation import test_accuracy

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
        
        # debug infos
        self.debug = False
    
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
        outputs = self(batch)
        labels = torch.tensor(labels).view(-1) # (batch*seq_len)
        
        cross_entropy_loss = nn.CrossEntropyLoss()
        loss = cross_entropy_loss(outputs.to("cpu"), labels)
        self.log_dict({"loss" : loss})
        return {'loss': loss}

    def predict(self, batch, filter_ids, labels):
        with torch.no_grad():
            outputs = self(batch)
            outputs = outputs.cpu().detach() # we don't need to compute gradient
            # we first need to filter-out the outputs relative to the labels != -100 (the one we are interested in)
            labels = torch.tensor(labels)
            mask = labels!=-100
            labels = labels[mask]
            labels_filter = mask.nonzero().view(-1) # returns the indeces to keep
            filter_outputs = torch.index_select(outputs, 0, labels_filter)
            ris = []
            for i in range(len(filter_outputs)):
                # predict only over the possible candidates
                filter_ids_i = torch.tensor(filter_ids[i])
                candidates_pred = torch.index_select(filter_outputs[i], 0, filter_ids_i)
                best_prediction = torch.argmax(candidates_pred, dim=0)
                ris.append(filter_ids_i[best_prediction.item()])
            return torch.tensor(ris), labels
    
    def validation_step(self, batch, batch_idx):
        labels = batch["cluster_gold"] if self.hparams.coarse_or_fine == "coarse" else batch["fine_gold"]
        outputs = self(batch)
        labels = torch.tensor(labels).view(-1)
        # LOSS
        cross_entropy_loss = nn.CrossEntropyLoss()
        val_loss = cross_entropy_loss(outputs.to("cpu"), labels)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        # ACCURACY
        filter_ids = batch["cluster_candidates"] if self.hparams.coarse_or_fine == "coarse" else batch["fine_candidates"]
        filter_ids = [l for item in filter_ids for l in item] # flattening
        labels = labels.tolist()
        preds, labels = self.predict(batch, filter_ids, labels) # output 'preds' and 'labels' both tensors
        assert preds.shape[0] == labels.shape[0]
        val_accuracy = test_accuracy(preds, labels)
        return {"val_accuracy": val_accuracy}
        
        # DEBUG info for one random item of the batch
        if self.debug:
            idx = random.randint(0, labels.shape[0]-1)
            filter_ids = [l for item in filter_ids for l in item]
            candidates = [self.id2sense[str(e)] for e in filter_ids[idx]]
            gold = self.id2sense[str(labels[idx].item())]
            pred = self.id2sense[str(preds[idx].item())]
            debug_infos = str(candidates) + " | " + str(gold) + " | " + str(pred)
            return {"val_accuracy": val_accuracy, "debug_infos" : debug_infos}
    
    # at the end of the epoch, we log the DEBUG infos of each batch
    def validation_epoch_end(self, outputs):
        # log average validation accuracy
        avg_val_accuracy = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("val_accuracy", avg_val_accuracy)
        
        if self.debug:
            c = [str(i+1) for i in range(len(outputs))]
            data = []
            for i in range(len(outputs)):
                data.append(outputs[i]["debug_infos"])
            self.logger.experiment.log({f"debug_infos_{self.current_epoch}": wandb.Table(columns=c, data=[data])})