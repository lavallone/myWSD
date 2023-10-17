import json
from tqdm import tqdm
import torch
import csv
import copy
import pickle
from src.data_module import read_dataset

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
        for batch in tqdm(data.test_dataloader()): ##!
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
def cluster_filter_evaluation(coarse_model, fine_model, data):
    # use the oracle or not
    oracle_or_not = False
    
    cluster2fine_map = json.load(open("data/mapping/cluster2fine_map.json", "r"))
    cluster_id2sense = json.load(open("data/mapping/cluster_id2sense.json", "r"))
    fine_sense2id = json.load(open("data/mapping/fine_sense2id.json", "r"))
    fine_model.eval()
    coarse_model.eval()
    with torch.no_grad():
        preds_list, labels_list = torch.tensor([]), torch.tensor([])
        for batch in tqdm(data.test_dataloader()): ##!
            if oracle_or_not == True: # if there's an oracle which tells us the correct homonym cluster
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
        for batch in tqdm(data.test_dataloader()): ##!
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

# COARSE evaluation of a subset of test/dev datasets using 4 different encoders type       
def coarse_subset_evaluation(model, data):
    items_id_list = []
    with open(data.hparams.data_test) as f: ##!
        data_dict = json.load(f)
    for sentence_id, sentence_data in list(data_dict.items()):
        # old structure
        if type(sentence_data["instance_ids"]) == dict:
            sense_idx_list = list(sentence_data["senses"].keys())
            for sense_idx in sense_idx_list:
                if data.hparams.cluster_candidates_filter and len(sentence_data["candidate_clusters"][sense_idx]) == 1:
                    continue
                items_id_list.append(sentence_data["instance_ids"][sense_idx])
        else: # new structure
            if data.hparams.cluster_candidates_filter and len(sentence_data["candidate_clusters"]) == 1:
                continue
            items_id_list.append(sentence_id)
    
    with open('data/subsets/test_dictionary.pkl', 'rb') as subset_file: ##!
        loaded_data = pickle.load(subset_file)
    subset_list = []
    for k in loaded_data.keys():
        if len(loaded_data[k]) == 0:
            subset_list.append(k)
        else:
            subset_list += loaded_data[k]
    subset_idx_list = []
    for i in range(len(items_id_list)):
        if items_id_list[i] in subset_list:
            subset_idx_list.append(i)
    
    # COARSE PREDICTIONS
    subset_idx_list = torch.tensor(subset_idx_list)
    model.eval()
    with torch.no_grad():
        preds_list, labels_list = torch.tensor([]), torch.tensor([])
        for batch in tqdm(data.test_dataloader()): ##!
            labels = batch["cluster_gold"]
            labels = [label for l in labels for label in l]
            candidates = batch["cluster_candidates"]
            candidates = [l for item in candidates for l in item]
            labels_eval = batch["cluster_gold_eval"]
            preds, labels = model.predict(batch, candidates, labels, labels_eval)
            assert preds.shape[0] == labels.shape[0]
            preds_list = torch.cat((preds_list, preds))
            labels_list = torch.cat((labels_list, labels))
            
        subset_preds_list = torch.index_select(preds_list, 0, subset_idx_list)
        subset_labels_list = torch.index_select(labels_list, 0, subset_idx_list)
        assert subset_preds_list.shape[0] == subset_labels_list.shape[0]
        print(f"\nOn a total of {subset_preds_list.shape[0]} samples...")
        ris_accuracy = test_accuracy(subset_preds_list, subset_labels_list)
        print()
        print(f"| Accuracy Score for test set:  {round(ris_accuracy, 4)} |")
        
# FINE2CLUSTER evaluation of a subset of test/dev datasets using 4 different encoders type       
def fine2cluster_subset_evaluation(model, data):
    items_id_list = []
    with open(data.hparams.data_test) as f: ##!
        data_dict = json.load(f)
    for sentence_id, sentence_data in list(data_dict.items()):
        # old structure
        if type(sentence_data["instance_ids"]) == dict:
            sense_idx_list = list(sentence_data["senses"].keys())
            for sense_idx in sense_idx_list:
                if data.hparams.cluster_candidates_filter and len(sentence_data["candidate_clusters"][sense_idx]) == 1:
                    continue
                items_id_list.append(sentence_data["instance_ids"][sense_idx])
        else: # new structure
            if data.hparams.cluster_candidates_filter and len(sentence_data["candidate_clusters"]) == 1:
                continue
            items_id_list.append(sentence_id)
    
    with open('data/subsets/test_dictionary_lemma.pkl', 'rb') as subset_file: ##!
        loaded_data = pickle.load(subset_file)
    subset_list = []
    for k in loaded_data.keys():
        if len(loaded_data[k]) == 0:
            subset_list.append(k)
        else:
            subset_list += loaded_data[k]
    subset_idx_list = []
    for i in range(len(items_id_list)):
        if items_id_list[i] in subset_list:
            subset_idx_list.append(i)
    
    # FINE2CLUSTER PREDICTIONS
    # mapping
    cluster2fine_map = json.load(open("data/mapping/cluster2fine_map.json", "r"))
    fine_id2sense = json.load(open("data/mapping/fine_id2sense.json", "r"))
    cluster_id2sense = json.load(open("data/mapping/cluster_id2sense.json", "r"))
    subset_idx_list = torch.tensor(subset_idx_list)
    model.eval()
    with torch.no_grad():
        preds_list, labels_list = [], []
        for batch in tqdm(data.test_dataloader()): ##!
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
            
        subset_preds_list = torch.index_select(torch.tensor(preds_list), 0, subset_idx_list)
        subset_labels_list = torch.index_select(torch.tensor(labels_list), 0, subset_idx_list)
        assert subset_preds_list.shape[0] == subset_labels_list.shape[0]
        print(f"\nOn a total of {subset_preds_list.shape[0]} samples...")
        ris_accuracy = test_accuracy(subset_preds_list, subset_labels_list)
        print()
        print(f"| Accuracy Score for test set:  {round(ris_accuracy, 4)} |")
        
        

########################################################################################      
# LOGGING FUNCTIONS FOR QUALITATIVELY EVALUATE fine2cluster AND cluster_filter METHODS #
########################################################################################

def log_fine2cluster(coarse_model, fine_model, data):
    
    coarse_csv_data_list = []
    fine2cluster_csv_data_list = []
    ids, sentences, senses = read_dataset(data.hparams.data_test)
    for i, sentence in enumerate(tqdm(sentences)):
        current_data_senses = senses[i]
        sense_idx_list = list( current_data_senses["cluster_gold"].keys() )
        for sense_idx in sense_idx_list:
            coarse_csv_data = [None, None, None, None, None]
            fine2cluster_csv_data = [None, None, None, None, None, None]
            coarse_csv_data[0] = ids[i]
            fine2cluster_csv_data[0] = ids[i]
            input_sentence = copy.deepcopy(sentence)
            input_sentence[int(sense_idx)] = "<<<" + input_sentence[int(sense_idx)] + ">>>" # we hihglight the word to disambiguate
            input_sentence = ' '.join(input_sentence) # from a  list of tokens to a string
            coarse_csv_data[1] = input_sentence
            fine2cluster_csv_data[1] = input_sentence
            # these below are all lists
            cluster_gold = [e for e in current_data_senses["cluster_gold"][sense_idx]]
            cluster_candidates = [e for e in current_data_senses["cluster_candidates"][sense_idx]]
            coarse_csv_data[3] = cluster_gold
            coarse_csv_data[4] = cluster_candidates
            fine2cluster_csv_data[4] = cluster_gold
            fine2cluster_csv_data[5] = cluster_candidates
            # if the data filtering is ON and the sense has only one cluster candidate, we skip it
            if data.hparams.cluster_candidates_filter and len(cluster_candidates) == 1:
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
            preds, _ = coarse_model.predict(batch, candidates, labels, labels_eval)
            preds = [cluster_id2sense[str(e)] for e in preds.tolist()]
            preds_list += preds
    assert len(preds_list) == len(coarse_csv_data_list)
    ############################################################################
    
    # FINE2CLUSTER MODEL PREDICTIONS
    ############################################################################
    fine_model.eval()
    with torch.no_grad():
        fine_preds_list, coarse_preds_list = [], []
        for batch in tqdm(data.test_dataloader()):
            # we first predict fine senses
            fine_labels = batch["fine_gold"]
            fine_labels = [label for l in fine_labels for label in l]
            fine_candidates = batch["fine_candidates"]
            fine_candidates = [l for item in fine_candidates for l in item]
            fine_labels_eval = batch["fine_gold_eval"]
            fine_preds, _ = fine_model.predict(batch, fine_candidates, fine_labels, fine_labels_eval)
            fine_preds_list += [fine_id2sense[str(e)] for e in fine_preds.tolist()]
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
            coarse_preds_list += [cluster_id2sense[str(e)] for e in coarse_preds]
    assert len(fine_preds_list) == len(fine2cluster_csv_data_list)
    assert len(coarse_preds_list) == len(fine2cluster_csv_data_list)
    ############################################################################
    
    coarse_correct_idx_list = []
    for i,l in enumerate(coarse_csv_data_list):
        if preds_list[i] in l[3]:
            l[2] = preds_list[i]
            coarse_correct_idx_list.append(i)
    print(f"\nTotal of {len(coarse_csv_data_list)} items")
    print(f"\nCOARSE model accuracy is {round(len(coarse_correct_idx_list)/len(coarse_csv_data_list),4)}")
            
    fine2cluster_correct_idx_list = []
    for i,l in enumerate(fine2cluster_csv_data_list):
        if coarse_preds_list[i] in l[4]:
            l[2] = fine_preds_list[i]
            l[3] = coarse_preds_list[i]
            fine2cluster_correct_idx_list.append(i)
    print(f"\nFINE2CLUSTER model accuracy is {round(len(fine2cluster_correct_idx_list)/len(fine2cluster_csv_data_list),4)}")
            
    # first log file --> instances correctly detected by coarse and not by fine2cluster
    ris_coarse_csv_data_list = []
    for coarse_idx in coarse_correct_idx_list:
        if coarse_idx not in fine2cluster_correct_idx_list:
            ris_coarse_csv_data_list.append(coarse_csv_data_list[coarse_idx])
    with open("log/log_coarse_yes_fine2cluster_no.csv", mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["ID", "SENTENCE", "COARSE_PRED", "COARSE_LABELS", "COARSE_CANDIDATES"])
        for csv_data in ris_coarse_csv_data_list:
            csv_writer.writerow(csv_data)
            
    # second log file --> instances correctly detected by fine2cluster and not by coarse
    ris_fine2cluster_csv_data_list = []
    for fine2cluster_idx in fine2cluster_correct_idx_list:
        if fine2cluster_idx not in coarse_correct_idx_list:
            ris_fine2cluster_csv_data_list.append(fine2cluster_csv_data_list[fine2cluster_idx])
    with open("log/log_fine2cluster_yes_coarse_no.csv", mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["ID", "SENTENCE", "FINE_PRED", "COARSE_PRED", "COARSE_LABELS", "COARSE_CANDIDATES"])
        for csv_data in ris_fine2cluster_csv_data_list:
            csv_writer.writerow(csv_data)
    

def log_cluster_filter(coarse_model, fine_model, data):
    
    fine_csv_data_list = []
    cluster_filter_csv_data_list = []
    ids, sentences, senses = read_dataset(data.hparams.data_test)
    for i, sentence in enumerate(sentences):
        current_data_senses = senses[i]
        sense_idx_list = list( current_data_senses["cluster_gold"].keys() )
        for sense_idx in sense_idx_list:
            fine_csv_data = [None, None, None, None, None]
            cluster_filter_csv_data = [None, None, None, None, None, None, None]
            fine_csv_data[0] = ids[i]
            cluster_filter_csv_data[0] = ids[i]
            input_sentence = copy.deepcopy(sentence)
            input_sentence[int(sense_idx)] = "<<<" + input_sentence[int(sense_idx)] + ">>>" # we hihglight the word to disambiguate
            input_sentence = ' '.join(input_sentence) # from a  list of tokens to a string
            fine_csv_data[1] = input_sentence
            cluster_filter_csv_data[1] = input_sentence
            # these below are all lists
            fine_gold = [e for e in current_data_senses["fine_gold"][sense_idx]]
            fine_candidates = [e for e in current_data_senses["fine_candidates"][sense_idx]]
            fine_csv_data[3] = fine_gold
            fine_csv_data[4] = fine_candidates
            cluster_filter_csv_data[5] = fine_gold
            cluster_filter_csv_data[6] = fine_candidates
            # if the data filtering is ON and the sense has only one cluster candidate, we skip it
            if data.hparams.cluster_candidates_filter and len([e for e in current_data_senses["cluster_candidates"][sense_idx]]) == 1:
                continue
            # otherwise we append
            fine_csv_data_list.append(fine_csv_data)
            cluster_filter_csv_data_list.append(cluster_filter_csv_data)
    
    # mapping
    cluster2fine_map = json.load(open("data/mapping/cluster2fine_map.json", "r"))
    cluster_id2sense = json.load(open("data/mapping/cluster_id2sense.json", "r"))
    fine_id2sense = json.load(open("data/mapping/fine_id2sense.json", "r"))
    fine_sense2id = json.load(open("data/mapping/fine_sense2id.json", "r"))
    
    # FINE MODEL PREDICTIONS
    ############################################################################
    fine_model.eval()
    with torch.no_grad():
        preds_list = []
        for batch in tqdm(data.test_dataloader()):
            labels = batch["fine_gold"]
            labels = [label for l in labels for label in l]
            candidates = batch["fine_candidates"]
            candidates = [l for item in candidates for l in item]
            labels_eval = batch["fine_gold_eval"]
            preds, _ = fine_model.predict(batch, candidates, labels, labels_eval)
            preds = [fine_id2sense[str(e)] for e in preds.tolist()]
            preds_list += preds
    assert len(preds_list) == len(fine_csv_data_list)
    ############################################################################
    
    # CLUSTER_FILTER MODEL PREDICTIONS
    ############################################################################
    oracle_or_not = True
    fine_model.eval()
    coarse_model.eval()
    with torch.no_grad():
        coarse_preds_list, fine_filtered_candidates_list, fine_preds_list = [], [], []
        for batch in tqdm(data.test_dataloader()):
            if oracle_or_not == True:
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
            coarse_preds_list += [cluster_id2sense[str(e)] for e in coarse_preds]
            
            # we now need the list of the fine senses within each predicted cluster
            filtered_fine_senses_list = []
            for pred_cluster in coarse_preds:
                # this is the list of fine senses (in indices)
                filtered_fine_senses = [ int(fine_sense2id[fine_sense[0]]) for fine_sense in cluster2fine_map[cluster_id2sense[str(pred_cluster)]] ]
                filtered_fine_senses_list.append(filtered_fine_senses)
            fine_filtered_candidates = []
            for l in filtered_fine_senses_list:
                new_l = []
                for e in l: 
                    new_l.append(fine_id2sense[str(e)])
                fine_filtered_candidates.append(new_l)
            fine_filtered_candidates_list += fine_filtered_candidates
            
            # we finally predict fine senses using the list just computed...
            fine_labels = batch["fine_gold"]
            fine_labels = [label for l in fine_labels for label in l] 
            fine_labels_eval = batch["fine_gold_eval"]
            fine_preds, _ = fine_model.predict(batch, filtered_fine_senses_list, fine_labels, fine_labels_eval)
            fine_preds_list += [fine_id2sense[str(e)] for e in fine_preds.tolist()]
    assert len(coarse_preds_list) == len(cluster_filter_csv_data_list)
    assert len(fine_filtered_candidates_list) == len(cluster_filter_csv_data_list)
    assert len(fine_preds_list) == len(cluster_filter_csv_data_list)
    ############################################################################
    
    fine_correct_idx_list = []
    for i,l in enumerate(fine_csv_data_list):
        if preds_list[i] in l[3]:
            l[2] = preds_list[i]
            fine_correct_idx_list.append(i)
    print(f"\nTotal of {len(fine_csv_data_list)} items")
    print(f"\nFINE model accuracy is {round(len(fine_correct_idx_list)/len(fine_csv_data_list),4)}")
            
    cluster_filter_correct_idx_list = []
    for i,l in enumerate(cluster_filter_csv_data_list):
        if fine_preds_list[i] in l[5]:
            l[2] = coarse_preds_list[i]
            l[3] = fine_filtered_candidates_list[i]
            l[4] = fine_preds_list[i]
            cluster_filter_correct_idx_list.append(i)
    print(f"\nCLUSTER_FILTER model accuracy is {round(len(cluster_filter_correct_idx_list)/len(cluster_filter_csv_data_list),4)}")
            
    # first log file --> instances correctly detected by coarse and not by fine2cluster
    ris_fine_csv_data_list = []
    for fine_idx in fine_correct_idx_list:
        if fine_idx not in cluster_filter_correct_idx_list:
            ris_fine_csv_data_list.append(fine_csv_data_list[fine_idx])
    csv_name_1 = "log/log_fine_yes_cluster_filter_"+"ORACLE"+"_no.csv" if oracle_or_not else "log/log_fine_yes_cluster_filter_no.csv"
    with open(csv_name_1, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["ID", "SENTENCE", "FINE_PRED", "FINE_LABELS", "FINE_CANDIDATES"])
        for csv_data in ris_fine_csv_data_list:
            csv_writer.writerow(csv_data)
            
    # second log file --> instances correctly detected by fine2cluster and not by coarse
    ris_cluster_filter_csv_data_list = []
    for cluster_filter_idx in cluster_filter_correct_idx_list:
        if cluster_filter_idx not in fine_correct_idx_list:
            ris_cluster_filter_csv_data_list.append(cluster_filter_csv_data_list[cluster_filter_idx])
    csv_name_2 = "log/log_cluster_filter_"+"ORACLE"+"_yes_fine_no.csv" if oracle_or_not else "log/log_cluster_filter_yes_fine_no.csv"
    with open(csv_name_2, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["ID", "SENTENCE", "COARSE_PRED", "FILTERED_FINE_CANDIDATES", "FINE_PRED", "FINE_LABELS", "FINE_CANDIDATES"])
        for csv_data in ris_cluster_filter_csv_data_list:
            csv_writer.writerow(csv_data)