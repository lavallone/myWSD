import json
from tqdm import tqdm
import torch

def test_accuracy(preds, labels):
    tot_n = preds.shape[0]
    correct_n = torch.sum((preds==labels)).item()
    return correct_n/tot_n

# TO RE-DO!!!!
# using a fine model, predict the homonym cluster of the word
def fine2cluster_evaluation(model, data):
    cluster2fine_map = json.load(open("data/mapping/cluster2fine_map.json", "r"))
    fine_id2sense = json.load(open("data/mapping/fine_id2sense.json", "r"))
    cluster_id2sense = json.load(open("data/mapping/cluster_id2sense.json", "r"))
    model.eval()
    with torch.no_grad():
        preds_list, labels_list = torch.tensor([]), torch.tensor([])
        for batch in tqdm(data.test_dataloader()):
            coarse_preds = torch.tensor([])
            coarse_labels = batch["cluster_gold"]
            coarse_labels = coarse_labels.cpu().detach()
            # we first predict fine senses
            fine_labels = batch["fine_gold"]
            fine_labels = fine_labels.cpu().detach()
            fine_preds, fine_labels = model.predict(batch, batch["fine_candidates"], fine_labels)
            # and then we find the correspondent 'homonym cluster'
            for fine_pred, cluster_candidates in zip(fine_preds, batch["cluster_candidates"]):
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
            # to handle the multi-label setting and being able to use Accuracy later...
            coarse_labels = model.manipulate_labels(coarse_labels, coarse_preds)
            assert len(coarse_preds) == len(coarse_labels)
        
            preds_list += coarse_preds
            labels_list += coarse_labels
        
        assert len(preds_list) == len(labels_list)
        print(f"\nOn a total of {len(preds_list)} samples...")
        ris_accuracy = test_accuracy(torch.tensor(preds_list), torch.tensor(labels_list)).item()
        print()
        print(f"| Accuracy Score for test set:  {round(ris_accuracy,4)} |")

# TO RE-DO!!!!   
# using a coarse-model to filter-out the set of possible fine sense predictions
def cluster_filter_evaluation(coarse_model, fine_model, data, oracle_or_not=False):
    cluster2fine_map = json.load(open("data/mapping/cluster2fine_map.json", "r"))
    cluster_id2sense = json.load(open("data/mapping/cluster_id2sense.json", "r"))
    fine_sense2id = json.load(open("data/mapping/fine_sense2id.json", "r"))
    fine_model.eval()
    coarse_model.eval()
    with torch.no_grad():
        preds_list, labels_list = [], []
        for batch in tqdm(data.test_dataloader()):
            # we save fine gold labels
            fine_labels = batch["fine_gold"]
            if oracle_or_not == True: # if there's an oracle which tells us the correct homonym cluster...
                coarse_gold_list = batch["cluster_gold"]
                # we took the decision to take into account the first right cluster (if there are more than one gold clusters)
                coarse_preds = [e[0] for e in coarse_gold_list]
            else:
                # we make cluster predictions
                coarse_preds = coarse_model.predict(batch, batch["cluster_candidates"])
            
            # we now need the list of the fine senses within each predicted cluster
            filtered_fine_senses_list = []
            for pred_cluster in coarse_preds:
                # this is the list of fine senses (in indices)
                filtered_fine_senses = [ int(fine_sense2id[fine_sense[0]]) for fine_sense in cluster2fine_map[cluster_id2sense[str(pred_cluster)]] ]
                filtered_fine_senses_list.append(filtered_fine_senses)
            
            # we finally predict fine senses using the list just computed... 
            fine_preds = fine_model.predict(batch, filtered_fine_senses_list)
            # manipulation of labels because of their multi-label nature
            fine_labels = fine_model.manipulate_labels(fine_labels, fine_preds)
            assert len(fine_labels) == len(fine_preds)
        
            preds_list += fine_preds
            labels_list += fine_labels
        
        assert len(preds_list) == len(labels_list)
        print(f"\nOn a total of {len(preds_list)} samples...")
        ris_accuracy = test_accuracy(torch.tensor(preds_list), torch.tensor(labels_list)).item()
        print()
        print(f"| Accuracy Score for test set:  {round(ris_accuracy,4)} |")
        
        
def base_evaluation(model, data):
    model.eval()
    with torch.no_grad():
        preds_list, labels_list = torch.tensor([]), torch.tensor([])
        for batch in tqdm(data.test_dataloader()):
            labels = batch["cluster_gold"] if self.hparams.coarse_or_fine == "coarse" else batch["fine_gold"]
            labels = [label for l in labels for label in l]
            candidates = batch["cluster_candidates"] if model.hparams.coarse_or_fine == "coarse" else batch["fine_candidates"]
            preds, labels = model.predict(batch, candidates, labels)
            assert preds.shape[0] == labels.shape[0]
            preds_list = torch.cat((preds_list, preds))
            labels_list = torch.cat((labels_list, labels))
        
        assert preds_list.shape[0] == labels_list.shape[0]
        print(f"\nOn a total of {len(preds_list.shape[0])} samples...")
        ris_accuracy = test_accuracy(preds_list, labels_list)
        print()
        print(f"| Accuracy Score for test set:  {round(ris_accuracy, 4)} |")