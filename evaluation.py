import json
from tqdm import tqdm
import torch

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
            fine_preds, _ = model.predict(batch, fine_candidates, fine_labels)
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

# TO BE REVISED...
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
                coarse_preds, _ = coarse_model.predict(batch, cluster_candidates, coarse_labels)
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
            fine_preds, fine_labels = fine_model.predict(batch, filtered_fine_senses_list, fine_labels)
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
            preds, labels = model.predict(batch, candidates, labels)
            assert preds.shape[0] == labels.shape[0]
            preds_list = torch.cat((preds_list, preds))
            labels_list = torch.cat((labels_list, labels))
        
        assert preds_list.shape[0] == labels_list.shape[0]
        print(f"\nOn a total of {preds_list.shape[0]} samples...")
        ris_accuracy = test_accuracy(preds_list, labels_list)
        print()
        print(f"| Accuracy Score for test set:  {round(ris_accuracy, 4)} |")