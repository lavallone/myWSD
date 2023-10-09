import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import json
from transformers import BertTokenizerFast

######################################### UTILITY PREPROCESSING FUNCTIONS ##############################################
## CLEAN TOKENS
def clean_tokens(data): # very simple token cleaner (not needed to make big operations here because BERT encoder works pretty well!)
    for sample in data:
        for i in range(len(sample["words"])):
            sample["words"][i] = sample["words"][i].lower()
            sample["words"][i] = sample["words"][i].replace(" ", "")
            sample["words"][i] = sample["words"][i].replace("`", "'")
            sample["words"][i] = sample["words"][i].replace("[", "(")
            sample["words"][i] = sample["words"][i].replace("]", ")")
            sample["words"][i] = sample["words"][i].encode("ascii", "ignore").decode()
        for i in range(len(sample["lemmas"])): # do it also for lemmas!
            sample["lemmas"][i] = sample["lemmas"][i].lower()
            sample["lemmas"][i] = sample["lemmas"][i].replace(" ", "")
            sample["lemmas"][i] = sample["lemmas"][i].replace("`", "'")
            sample["lemmas"][i] = sample["lemmas"][i].replace("[", "(")
            sample["lemmas"][i] = sample["lemmas"][i].replace("]", ")")
            sample["lemmas"][i] = sample["lemmas"][i].encode("ascii", "ignore").decode()
                
## FILTER SENTENCES
def filter_sentences(train_sentences, train_senses, min_sent_length=5, max_sent_length=85):
    train_items = list(zip(train_sentences, train_senses))
    train_items = list(filter(lambda x: len(x[0]["words"])>=min_sent_length and len(x[0]["words"])<=max_sent_length, train_items)) # min and max length
    for item in train_items:
        # we check that each train sentence has at least one word to be disambiguated!
        assert len(item[0]["instance_ids"].keys()) != 0
    ris1, ris2 = [], []
    for e1,e2 in train_items:
        ris1.append(e1)
        ris2.append(e2)
    return ris1, ris2

########################################################################################################################                

# utility function for reading the dataset
def read_dataset(path):
    sentences_list, senses_list = [], []
    with open(path) as f:
        data = json.load(f)
    for sentence_data in list(data.values()):
        assert len(sentence_data["instance_ids"]) > 0
        assert len(sentence_data["words"]) == len(sentence_data["lemmas"]) == len(sentence_data["pos_tags"])
        sentences_list.append(sentence_data["words"])
        
        assert (len(sentence_data["instance_ids"]) ==
                len(sentence_data["gold_clusters"]) ==
                len(sentence_data["candidate_clusters"]) ==
                len(sentence_data["senses"]) ==
                len(sentence_data["wn_candidates"])
                )
        assert all(len(gt) > 0 for gt in sentence_data["gold_clusters"].values())
        assert (all(gt_sense in candidates for gt_sense in gt)
            for gt, candidates in zip(sentence_data["gold_clusters"].values(), sentence_data["candidate_clusters"].values()))
        assert all(len(gt) > 0 for gt in sentence_data["senses"].values())
        assert (all(gt_sense in candidates for gt_sense in gt)
                for gt, candidates in zip(sentence_data["senses"].values(), sentence_data["wn_candidates"].values()))
        senses = {}
        # COARSE
        senses["cluster_gold"] = sentence_data["gold_clusters"]
        senses["cluster_candidates"] = sentence_data["candidate_clusters"]
        # FINE
        senses["fine_gold"] = sentence_data["senses"]
        senses["fine_candidates"] = sentence_data["wn_candidates"]
        senses_list.append(senses)
        
    assert len(sentences_list) == len(senses_list)
    return sentences_list, senses_list

# function deals with multi-label items and proceeds in this way:
# it select as the only one gold label the one which appears first in the candidates set!
def manipulate_labels_1(gold, candidates):
    candidates = list(filter(lambda x: x in gold, candidates))
    return candidates[0] # the first occurrence

# function sets the right labels and indeed the -100 values to the tokenized sequence
# phase needed to speed-up computation and to be used by CrossEntropy loss 
def manipulate_labels_2(sense_ids_list, word_ids_list, labels_list):
    labels = []
    for i,word_ids in enumerate(word_ids_list):
        labels_ids = []
        for word_id in word_ids:
            if word_id is None: # special tokens case
                labels_ids.append(-100)
            elif word_id in sense_ids_list[i]:
                if word_id != previous_word_id:
                    labels_ids.append(labels_list[i][sense_ids_list[i].index(word_id)])
                else: # if the token is not the first one
                    labels_ids.append(-100)
            else: # if it's not a word to disambiguate
                labels_ids.append(-100)
            previous_word_id = word_id
        labels.append(labels_ids)
        
    return labels # list of lists
     

class WSD_Dataset(Dataset):
    def __init__(self, data_sentences, data_senses, coarse_sense2id_path, fine_sense2id_path):
        self.data = list()
        self.data_sentences = data_sentences # list of input sentences
        self.data_senses = data_senses # list of dictionaries relative to each input sentence
        self.coarse_sense2id = json.load(open(coarse_sense2id_path, "r"))
        self.fine_sense2id = json.load(open(fine_sense2id_path, "r"))
        self.make_data()
    
    def make_data(self):
        for i, sentence in enumerate(self.data_sentences):
            input_sentence = sentence # list of tokens
            current_data_senses = self.data_senses[i]
            sense_idx_list = list( current_data_senses["cluster_gold"].keys() )
            cluster_gold_list, cluster_candidates_list, fine_gold_list, fine_candidates_list = [], [], [], []
            for sense_idx in sense_idx_list:
                # these below are all lists
                cluster_gold = [self.coarse_sense2id[e] for e in current_data_senses["cluster_gold"][sense_idx]]
                cluster_candidates = [self.coarse_sense2id[e] for e in current_data_senses["cluster_candidates"][sense_idx]]
                fine_gold = [self.fine_sense2id[e] for e in current_data_senses["fine_gold"][sense_idx]]
                fine_candidates = [self.fine_sense2id[e] for e in current_data_senses["fine_candidates"][sense_idx]]
                
                cluster_gold = manipulate_labels_1(cluster_gold, cluster_candidates)
                cluster_gold_list.append(cluster_gold) # list
                cluster_candidates_list.append(cluster_candidates) # list of lists
                fine_gold = manipulate_labels_1(fine_gold, fine_candidates)
                fine_gold_list.append(fine_gold) # list
                fine_candidates_list.append(fine_candidates) # list of lists
            # an item for each sentence
            self.data.append({"sense_ids" : [int(sense_idx) for sense_idx in sense_idx_list], "input": input_sentence, "cluster_gold_list" : cluster_gold_list, "cluster_candidates_list" : cluster_candidates_list, "fine_gold_list" : fine_gold_list, "fine_candidates_list" : fine_candidates_list})
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class WSD_DataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams, logger=False)
        
        self.train_sentences, self.train_senses = read_dataset(self.hparams.data_train)
        self.val_sentences, self.val_senses = read_dataset(self.hparams.data_val)
        self.test_sentences, self.test_senses = read_dataset(self.hparams.data_test)

    def setup(self, stage=None):
        # TRAIN
        #clean_tokens(self.train_sentences)
        #self.train_sentences, self.train_senses = filter_sentences(self.train_sentences, self.train_senses)
        self.data_train = WSD_Dataset(data_sentences=self.train_sentences, data_senses=self.train_senses, coarse_sense2id_path="data/mapping/cluster_sense2id.json", fine_sense2id_path="data/mapping/fine_sense2id.json")
        # VAL
        #clean_tokens(self.val_sentences)
        self.data_val = WSD_Dataset(data_sentences=self.val_sentences, data_senses=self.val_senses, coarse_sense2id_path="data/mapping/cluster_sense2id.json", fine_sense2id_path="data/mapping/fine_sense2id.json")
        # TEST
        #clean_tokens(self.test_sentences)
        self.data_test = WSD_Dataset(data_sentences=self.test_sentences, data_senses=self.test_senses, coarse_sense2id_path="data/mapping/cluster_sense2id.json", fine_sense2id_path="data/mapping/fine_sense2id.json")

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.n_cpu,
            collate_fn = self.collate,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.n_cpu,
            collate_fn = self.collate,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.n_cpu,
            collate_fn = self.collate,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True
        )
    
    # for efficiency reasons, each time we pick a batch from the dataloader, we call this function!
    def collate(self, batch):
        batch_out = dict()
        tokenizer = BertTokenizerFast.from_pretrained("bert-large-cased")
        batch_out["inputs"] = tokenizer([sample["input"] for sample in batch], padding=True, truncation=True, return_tensors="pt", is_split_into_words=True)
        # to check if no sequence is being truncated
        assert len(batch_out["inputs"]["input_ids"]) < 1024
        
        sense_ids_list = [sample["sense_ids"] for sample in batch] # list of lists of lists
        cluster_gold_list = [sample["cluster_gold_list"] for sample in batch] # list of lists
        batch_out["cluster_gold"] = manipulate_labels_2(sense_ids_list, [batch_out["inputs"].word_ids(i) for i in range(len(batch))], cluster_gold_list)
        batch_out["cluster_candidates"] = [sample["cluster_candidates_list"] for sample in batch] # list of lists of lists
        fine_gold_list = [sample["fine_gold_list"] for sample in batch] # list of lists
        batch_out["fine_gold"] = manipulate_labels_2(sense_ids_list, [batch_out["inputs"].word_ids(i) for i in range(len(batch))], fine_gold_list)
        batch_out["fine_candidates"] = [sample["fine_candidates_list"] for sample in batch] # list of lists of lists
        return batch_out