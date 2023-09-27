import torch
from torch import optim
from torch import nn
import pytorch_lightning as pl
from transformers import BertModel
from torchmetrics import Accuracy
import json

class WSD_Model(pl.LightningModule):
    def __init__(self, hparams):
        super(WSD_Model, self).__init__()
        self.save_hyperparameters(hparams)
        print(f"___________ {self.hparams.coarse_or_fine} MODEL ___________\n")
        self.encoder = BertModel.from_pretrained("bert-large-uncased") # there's the possibility to not download it each time!
        
        # we set all parameters to be not trainable
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # classification HEAD
        self.batch_norm = nn.BatchNorm1d(4096)
        self.hidden_MLP = nn.Linear(4096, self.hparams.hidden_dim, bias=True)
        self.act_fun = nn.SiLU(inplace=True) # Swish activation function
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.num_senses = 106880 if self.hparams.coarse_or_fine == "coarse" else 117659
        self.classifier = nn.Linear(self.hparams.hidden_dim, self.num_senses, bias=False) # final linear projection with no bias
        
        # mapping from 'id' to name of the 'sense'
        #self.id2sense = json.load(open("data/mapping/cluster_id2sense.json", "r")) if self.hparams.coarse_or_fine == "coarse" else json.load(open("data/mapping/fine_id2sense.json", "r"))
        
        # validation ACCURACY
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_senses)
       
               
    def forward(self, batch):
        text = batch["inputs"]
        embed_text = self.encoder(text["input_ids"].to(self.device), attention_mask=text["attention_mask"].to(self.device), token_type_ids=text["token_type_ids"].to(self.device), output_hidden_states=True)
        # I take the hidden representation of the last four layers of each token
        #embed_text = torch.stack(embed_text.hidden_states[-4:], dim=0).sum(dim=0) # sum
        #embed_text = torch.stack(embed_text.hidden_states[-4:], dim=0).sum(dim=0) # mean
        embed_text = torch.cat(embed_text.hidden_states[-4:], dim=-1) # concatenation --> (batch, 1024*4=4096)
        
        # I select the FIRST embedding of the word we want to disambiguate (first I did the average)
        encoder_output_list = []
        for i in range(len(batch["sense_ids"])):
            first_idx = int(batch["sense_ids"][i][0])
            #last_idx = int(batch["sense_ids"][i][-1] + 1)
            #select_word_embs = embed_text[i, first_idx:last_idx, :]
            #word_emb = select_word_embs.mean(dim=0)
            word_emb = embed_text[i, first_idx, :] # I only took the first token embedding
            encoder_output_list.append(word_emb)
        encoder_output = torch.stack(encoder_output_list, dim=0) # (batch, 4096)
        
        encoder_output_norm = self.batch_norm(self.dropout(encoder_output))
        hidden_output = self.dropout(self.act_fun(self.hidden_MLP(encoder_output_norm)))
        
        return self.classifier(hidden_output)

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=self.hparams.lr)
        return {
            "optimizer": optimizer,
        }

    # CROSS-ENTROPY
    def loss_function(self, outputs, labels):
        loss_function = nn.BCEWithLogitsLoss()
        # one-hot encoding for the gold senses (multi-label problem)
        one_hot_encoding_labels = torch.zeros([outputs.shape[0], self.num_senses], dtype=torch.float32).to(self.device)
        for i,l in enumerate(labels):
            one_hot_encoding_labels[i, l] = 1
        loss = loss_function(outputs, one_hot_encoding_labels)
        return loss
    
    def training_step(self, batch, batch_idx):
        labels = batch["cluster_gold"] if self.hparams.coarse_or_fine == "coarse" else batch["fine_gold"]
        outputs = self(batch) # (not normalized)
        
        loss = self.loss_function(outputs, labels)
        self.log_dict({"loss" : loss})
        # since we only monitor the loss for the training phase, we don't need to write additional 
        # code in the 'training_epoch_end' function!
        return {'loss': loss}

    # provided by 'filter_ids' argument for future use-cases
    def predict(self, batch, filter_ids):
        with torch.no_grad():
            outputs = self(batch)
            ris = []
            for i in range(len(outputs)):
                # predict only over the possible homonym clusters...
                candidates_pred = torch.index_select(outputs[i], 0, torch.tensor(filter_ids[i]).to(self.device))
                best_prediction = torch.argmax(candidates_pred, dim=0)
                ris.append(filter_ids[i][best_prediction.item()])
            return ris # list of predicted senses (expressed in indices)
    
    # this is needed because the words can actually have more than one gold sense.
    # Indeed, in our case, the prediction is correct if belongs to one of the gold senses! 
    def manipulate_labels(self, labels, preds):
        ris_labels = []
        for i,pred in enumerate(preds):
            if pred in labels[i]:
                ris_labels.append(pred)
            else:
                ris_labels.append(labels[i][0])
        return ris_labels
    
    def validation_step(self, batch, batch_idx):
        labels = batch["cluster_gold"] if self.hparams.coarse_or_fine == "coarse" else batch["fine_gold"]
        outputs = self(batch)
        # LOSS
        val_loss = self.loss_function(outputs, labels)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        # ACCURACY
        filter_ids = batch["cluster_candidates"] if self.hparams.coarse_or_fine == "coarse" else batch["fine_candidates"]
        preds = self.predict(batch, filter_ids)
        labels = self.manipulate_labels(labels, preds)
        assert len(preds) == len(labels)
        self.accuracy.update(torch.tensor(preds), torch.tensor(labels))
        self.log("val_accuracy", self.accuracy, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)