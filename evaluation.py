import json
from tqdm import tqdm
from torchmetrics import Accuracy
import torch


def fine2cluster_evaluation(model, data):
    test_accuracy = Accuracy(task="multiclass", num_classes=model.num_senses, average="micro")
    cluster2fine_map = json.load(open("data/mapping/cluster2fine_map.json", "r"))
    fine_id2sense = json.load(open("data/mapping/fine_id2sense.json", "r"))
    cluster_id2sense = json.load(open("data/mapping/cluster_id2sense.json", "r"))
    model.eval()
    with torch.no_grad():
        preds_list, labels_list = [], []
        for batch in tqdm(data.test_dataloader()):
            coarse_preds = []
            coarse_labels = batch["cluster_gold"]
            # we first predict fine senses
            fine_preds = model.predict(batch, batch["fine_candidates"])
            # and then we find the correspondent 'homonym cluster'
            for fine_pred, cluster_candidates in zip(fine_preds, batch["cluster_candidates"]):
                cluster_found = False
                for cluster_candidate in cluster_candidates:
                    if cluster_found == True: break
                    fine_senses_list = cluster2fine_map[cluster_id2sense[str(cluster_candidate)]] # list of fine senses of a cluster
                    for fine_sense in fine_senses_list:
                        if cluster_found == True: break
                        if fine_id2sense[str(fine_pred)] == fine_sense[0]:
                            coarse_preds.append(cluster_candidate)
                            cluster_found = True
            coarse_labels = model.manipulate_labels(coarse_labels, coarse_preds)
            assert len(coarse_preds) == len(coarse_labels)
        
            preds_list += coarse_preds
            labels_list += coarse_labels
        
        assert len(preds_list) == len(labels_list)
        print(f"\nOn a total of {len(preds_list)} samples...")
        ris_accuracy = test_accuracy(torch.tensor(preds_list), torch.tensor(labels_list)).item()
        print()
        print(f"| Accuracy Score for test set:  {round(ris_accuracy,4)} |")
        
        
def cluster_filter_evaluation(coarse_model, fine_model, data, oracle_or_not=False):
    test_accuracy = Accuracy(task="multiclass", num_classes=fine_model.num_senses, average="micro")
    cluster2fine_map = json.load(open("data/mapping/cluster2fine_map.json", "r"))
    cluster_id2sense = json.load(open("data/mapping/cluster_id2sense.json", "r"))
    fine_sense2id = json.load(open("data/mapping/fine_sense2id.json", "r"))
    fine_model.eval()
    coarse_model.eval()
    with torch.no_grad():
        preds_list, labels_list = [], []
        for batch in tqdm(data.test_dataloader()):
            fine_labels = batch["fine_gold"]
            if oracle_or_not == True: # if there's an oracle which tells us the correct homonym cluster...
                coarse_gold_list = batch["cluster_gold"]
                # we took the decision to take into account the first right cluster (if there are more than one correct)
                coarse_preds = [e[0] for e in coarse_gold_list]
            else:
                coarse_preds = coarse_model.predict(batch, batch["cluster_candidates"])
            
            filtered_fine_senses_list = []
            for pred_cluster in coarse_preds:
                filtered_fine_senses = [ int(fine_sense2id[fine_sense[0]]) for fine_sense in cluster2fine_map[cluster_id2sense[str(pred_cluster)]] ]
                filtered_fine_senses_list.append(filtered_fine_senses)
                
            fine_preds = fine_model.predict(batch, filtered_fine_senses_list)
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
    test_accuracy = Accuracy(task="multiclass", num_classes=model.num_senses, average="micro")
    model.eval()
    with torch.no_grad():
        preds_list, labels_list = [], []
        for batch in tqdm(data.test_dataloader()):
            candidates = batch["cluster_candidates"] if model.hparams.coarse_or_fine == "coarse" else batch["fine_candidates"]
            preds = model.predict(batch, candidates)
            
            labels = batch["cluster_gold"] if model.hparams.coarse_or_fine == "coarse" else batch["fine_gold"]
            labels = model.manipulate_labels(labels, preds)
            assert len(labels) == len(preds)
            preds_list += preds
            labels_list += labels
        
        assert len(preds_list) == len(labels_list)
        print(f"\nOn a total of {len(preds_list)} samples...")
        ris_accuracy = test_accuracy(torch.tensor(preds_list), torch.tensor(labels_list)).item()
        print()
        print(f"| Accuracy Score for test set:  {round(ris_accuracy,4)} |")