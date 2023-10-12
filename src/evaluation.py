import json
from tqdm import tqdm
import torch
import csv

# compute accuracy
def test_accuracy(preds, labels):
    tot_n = preds.shape[0]
    correct_n = torch.sum((preds==labels)).item()
    return correct_n/tot_n

# using a fine model, predict the homonym cluster of the word
def fine2cluster_evaluation(model, data):
    cluster2fine_map = json.load(open("data/mapping/cluster2fine_map.json", "r"))
    fine_id2sense = json.load(open("data/mapping/fine_id2sense.json", "r"))
    cluster_id2sense = json.load(open("data/mapping/cluster_id2sense.json", "r"))
    model.eval()
    with torch.no_grad():
        preds_list, labels_list = [], []
        for batch in tqdm(data.test_dataloader()):
            # we first predict fine senses
            fine_labels = batch["fine_gold"]
            fine_labels = [label for l in fine_labels for label in l]
            fine_candidates = batch["fine_candidates"]
            fine_candidates = [l for item in fine_candidates for l in item]
            fine_labels_eval = batch["fine_gold_eval"]
            fine_preds, _ = model.predict(batch, fine_candidates, fine_labels, fine_labels_eval)
            # we prepare fine_preds and cluster_candidates...
            fine_preds = fine_preds.tolist()
            cluster_candidates_list = [e for l in batch["cluster_candidates"] for e in l]
            # and then we find the correspondent 'homonym cluster'
            coarse_preds = [] # to be filled
            for fine_pred, cluster_candidates in zip(fine_preds, cluster_candidates_list):
                cluster_found = False
                for cluster_candidate in cluster_candidates:
                    if cluster_found == True: break
                    # we compute the list of all fine sense in the 'cluster_candidate' homonym cluster
                    fine_senses_list = cluster2fine_map[cluster_id2sense[str(cluster_candidate)]] # list of fine senses of a cluster
                    for fine_sense in fine_senses_list:
                        if cluster_found == True: break
                        if fine_id2sense[str(fine_pred)] == fine_sense[0]: # because fine_sense[1] is the gloss of the fine sense
                            coarse_preds.append(cluster_candidate)
                            cluster_found = True
            # we prepare coarse labels
            coarse_labels = torch.tensor(batch["cluster_gold"])
            mask = coarse_labels!=-100
            coarse_labels = coarse_labels[mask]
            coarse_labels = coarse_labels.tolist() # already flattened
            assert len(coarse_preds) == len(coarse_labels)
        
            preds_list += coarse_preds
            labels_list += coarse_labels
        
        assert len(preds_list) == len(labels_list)
        print(f"\nOn a total of {len(preds_list)} samples...")
        ris_accuracy = test_accuracy(torch.tensor(preds_list), torch.tensor(labels_list))
        print()
        print(f"| Accuracy Score for test set:  {round(ris_accuracy, 4)} |")

# using a coarse-model to filter-out the set of possible fine sense predictions
def cluster_filter_evaluation(coarse_model, fine_model, data, oracle_or_not=False):
    cluster2fine_map = json.load(open("data/mapping/cluster2fine_map.json", "r"))
    cluster_id2sense = json.load(open("data/mapping/cluster_id2sense.json", "r"))
    fine_sense2id = json.load(open("data/mapping/fine_sense2id.json", "r"))
    fine_model.eval()
    coarse_model.eval()
    with torch.no_grad():
        preds_list, labels_list = torch.tensor([]), torch.tensor([])
        for batch in tqdm(data.test_dataloader()):
            if oracle_or_not == True: # if there's an oracle which tells us the correct homonym cluster...
                coarse_preds = torch.tensor(batch["cluster_gold"])
                mask = coarse_preds!=-100
                coarse_preds = coarse_preds[mask]
                coarse_preds = coarse_preds.tolist() # already flattened
            else:
                # we make cluster predictions
                coarse_labels = batch["cluster_gold"]
                coarse_labels = [label for l in coarse_labels for label in l]
                cluster_candidates = batch["cluster_candidates"]
                cluster_candidates = [l for item in cluster_candidates for l in item]
                coarse_labels_eval = batch["cluster_gold_eval"]
                coarse_preds, _ = coarse_model.predict(batch, cluster_candidates, coarse_labels, coarse_labels_eval)
                coarse_preds = coarse_preds.tolist()
            
            # we now need the list of the fine senses within each predicted cluster
            filtered_fine_senses_list = []
            for pred_cluster in coarse_preds:
                # this is the list of fine senses (in indices)
                filtered_fine_senses = [ int(fine_sense2id[fine_sense[0]]) for fine_sense in cluster2fine_map[cluster_id2sense[str(pred_cluster)]] ]
                filtered_fine_senses_list.append(filtered_fine_senses)
            
            # we finally predict fine senses using the list just computed...
            fine_labels = batch["fine_gold"]
            fine_labels = [label for l in fine_labels for label in l] 
            fine_labels_eval = batch["fine_gold_eval"]
            fine_preds, fine_labels = fine_model.predict(batch, filtered_fine_senses_list, fine_labels, fine_labels_eval)
            assert fine_labels.shape[0] == fine_preds.shape[0]
        
            preds_list = torch.cat((preds_list, fine_preds))
            labels_list = torch.cat((labels_list, fine_labels))
        
        assert labels_list.shape[0] == preds_list.shape[0]
        print(f"\nOn a total of {preds_list.shape[0]} samples...")
        ris_accuracy = test_accuracy(preds_list, labels_list)
        print()
        print(f"| Accuracy Score for test set:  {round(ris_accuracy, 4)} |")
        
        
def base_evaluation(model, data):
    model.eval()
    with torch.no_grad():
        preds_list, labels_list = torch.tensor([]), torch.tensor([])
        for batch in tqdm(data.test_dataloader()):
            labels = batch["cluster_gold"] if model.hparams.coarse_or_fine == "coarse" else batch["fine_gold"]
            labels = [label for l in labels for label in l]
            candidates = batch["cluster_candidates"] if model.hparams.coarse_or_fine == "coarse" else batch["fine_candidates"]
            candidates = [l for item in candidates for l in item]
            labels_eval = batch["cluster_gold_eval"] if model.hparams.coarse_or_fine == "coarse" else batch["fine_gold_eval"]
            preds, labels = model.predict(batch, candidates, labels, labels_eval)
            assert preds.shape[0] == labels.shape[0]
            preds_list = torch.cat((preds_list, preds))
            labels_list = torch.cat((labels_list, labels))
        
        assert preds_list.shape[0] == labels_list.shape[0]
        print(f"\nOn a total of {preds_list.shape[0]} samples...")
        ris_accuracy = test_accuracy(preds_list, labels_list)
        print()
        print(f"| Accuracy Score for test set:  {round(ris_accuracy, 4)} |")
        
# LOGGING FUNCTIONS FOR QUALITATIVELY EVALUATE fine2cluster AND cluster_filter METHODS

def log_fine2cluster(coarse_model, fine_model, data):
    
    coarse_csv_data_list = []
    fine2cluster_csv_data_list = []
    ids, sentences, senses = read_dataset(data.hparams.data_test)
    for i, sentence in tqdm(enumerate(sentences)):
        current_data_senses = self.data_senses[i]
        sense_idx_list = list( current_data_senses["cluster_gold"].keys() )
        for sense_idx in sense_idx_list:
            coarse_csv_data = [None, None, None, None, None]
            fine2cluster_csv_data = [None, None, None, None, None, None]
            coarse_csv_data[0] = ids[0]
            fine2cluster_csv_data[0] = ids[0]
            sentence[sense_idx] = "<<<" + sentence[sense_idx] + ">>>" # we hihglight the word to disambiguate
            sentence = ' '.join(sentence) # from a  list of tokens to a string
            coarse_csv_data[1] = sentence
            fine2cluster_csv_data[1] = sentence
            # these below are all lists
            cluster_gold = [e for e in current_data_senses["cluster_gold"][sense_idx]]
            cluster_candidates = [e for e in current_data_senses["cluster_candidates"][sense_idx]]
            coarse_csv_data[3] = cluster_gold
            coarse_csv_data[4] = cluster_candidates
            fine2cluster_csv_data[4] = cluster_gold
            fine2cluster_csv_data[5] = cluster_candidates
            # if the data filtering is ON and the sense has only one cluster candidate, we skip it
            if self.cluster_candidates_filter and len(cluster_candidates) == 1:
                continue
            # otherwise we append
            coarse_csv_data_list.append(coarse_csv_data)
            fine2cluster_csv_data_list.append(fine2cluster_csv_data)
    
    # mapping
    cluster2fine_map = json.load(open("data/mapping/cluster2fine_map.json", "r"))
    fine_id2sense = json.load(open("data/mapping/fine_id2sense.json", "r"))
    cluster_id2sense = json.load(open("data/mapping/cluster_id2sense.json", "r"))
    
    # COARSE MODEL PREDICTIONS
    ############################################################################
    coarse_model.eval()
    with torch.no_grad():
        preds_list = []
        for batch in tqdm(data.test_dataloader()):
            labels = batch["cluster_gold"]
            labels = [label for l in labels for label in l]
            candidates = batch["cluster_candidates"]
            candidates = [l for item in candidates for l in item]
            labels_eval = batch["cluster_gold_eval"]
            preds, _ = model.predict(batch, candidates, labels, labels_eval)
            preds = [cluster_id2sense[e] for e in preds.tolist()]
            preds_list.append(preds)
    assert len(preds_list) == len(coarse_csv_data_list)
    ############################################################################
    
    # FINE2CLUSTER MODEL PREDICTIONS
    ############################################################################
    model.eval()
    with torch.no_grad():
        fine_preds_list, coarse_preds_list = [], []
        for batch in tqdm(data.test_dataloader()):
            # we first predict fine senses
            fine_labels = batch["fine_gold"]
            fine_labels = [label for l in fine_labels for label in l]
            fine_candidates = batch["fine_candidates"]
            fine_candidates = [l for item in fine_candidates for l in item]
            fine_labels_eval = batch["fine_gold_eval"]
            fine_preds, _ = model.predict(batch, fine_candidates, fine_labels, fine_labels_eval)
            fine_preds_list.append([fine_id2sense[e] for e in fine_preds.tolist()])
            # we prepare fine_preds and cluster_candidates...
            fine_preds = fine_preds.tolist()
            cluster_candidates_list = [e for l in batch["cluster_candidates"] for e in l]
            # and then we find the correspondent 'homonym cluster'
            coarse_preds = [] # to be filled
            for fine_pred, cluster_candidates in zip(fine_preds, cluster_candidates_list):
                cluster_found = False
                for cluster_candidate in cluster_candidates:
                    if cluster_found == True: break
                    # we compute the list of all fine sense in the 'cluster_candidate' homonym cluster
                    fine_senses_list = cluster2fine_map[cluster_id2sense[str(cluster_candidate)]] # list of fine senses of a cluster
                    for fine_sense in fine_senses_list:
                        if cluster_found == True: break
                        if fine_id2sense[str(fine_pred)] == fine_sense[0]: # because fine_sense[1] is the gloss of the fine sense
                            coarse_preds.append(cluster_candidate)
                            cluster_found = True
            coarse_preds_list.append([cluster_id2sense[e] for e in coarse_preds.tolist()])
    assert len(fine_preds_list) == len(fine2cluster_csv_data_list)
    assert len(coarse_preds_list) == len(fine2cluster_csv_data_list)
    ############################################################################
    
    coarse_correct_idx_list = []
    for i,l in enumerate(coarse_csv_data_list):
        if preds_list[i] in l[3]:
            l[2] = preds_list[i]
            coarse_correct_idx_list.append(i)
            
    fine2cluster_correct_idx_list = []
    for i,l in enumerate(fine2cluster_csv_data_list):
        if coarse_preds_list[i] in l[4]:
            l[2] = fine_preds_list[i]
            l[3] = coarse_preds_list[i]
            fine2cluster_correct_idx_list.append(i)
            
    # first log file --> instances correctly detected by coarse and not by fine2cluster
    ris_coarse_csv_data_list = []
    for coarse_idx in coarse_correct_idx_list:
        if coarse_idx not in fine2cluster_correct_idx_list:
            ris_coarse_csv_data_list.append(coarse_csv_data_list[coarse_idx])
    with open("log_coarse_yes_fine2cluster_no.csv", mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for csv_data in ris_coarse_csv_data_list:
            csv_writer.writerow(csv_data)
            
    # second log file --> instances correctly detected by fine2cluster and not by coarse
    ris_fine2cluster_csv_data_list = []
    for fine2cluster_idx in fine2cluster_correct_idx_list:
        if fine2cluster_idx not in coarse_correct_idx_list:
            ris_fine2cluster_csv_data_list.append(fine2cluster_csv_data_list[fine2cluster_idx])
    with open("log_fine2cluster_yes_coarse_no.csv", mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for csv_data in ris_fine2cluster_csv_data_list:
            csv_writer.writerow(csv_data)
    
    
            
            