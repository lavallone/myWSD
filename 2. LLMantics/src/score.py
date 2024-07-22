from sklearn.metrics import precision_score, recall_score, f1_score
from typing import NamedTuple, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import argparse
import json
import nltk
import os

def _choose_definition(instance_gold, answer):
    """
    Chooses the most appropriate definition from a list of definitions for a given answer.
    
    Args:
        instance_gold (Dict): One instance of the golden data.
        answer (str): The answer to be disambiguated.
    
    Returns:
        str: The chosen definition with the highest overlap score with the answer.
    """
    id_ = instance_gold["id"]
    definitions = instance_gold["definitions"]
    
    if args.subtask == "generation":
        global id2vec_gold, id2vec_disambiguated_data

    # when we want to select sense keys, we first try to locate the key in the answer...
    # if we don't succeed, we use instead lexical overlap!
    if args.subtask == "selection" and args.prompt_type == "v3":
        tokenized_answer = list(nltk.word_tokenize(answer))
        for candidate, definition in list(zip(instance_gold["candidates"], definitions)):
            if candidate in tokenized_answer:
                return definition

    definition2overlap = {}
    for definition in definitions:
        if args.subtask == "generation":
            vec1, vec2 = id2vec_gold[id_+definition], id2vec_disambiguated_data[id_]
            overlap = _compute_semantic_overlap(vec1, vec2)
        else:
            overlap = _compute_lexical_overlap(definition, answer)
        definition2overlap[definition] = overlap
    return max(definition2overlap, key=definition2overlap.get)

def _compute_lexical_overlap(definition, answer):
    """
    Computes the overlap between two strings based on tokenized words.
    
    Args:
        definition (str): The first string to compare.
        answer (str): The second string to compare.
    
    Returns:
        float: The overlap score between the two strings, 
               computed as the ratio of the number of common tokens
               to the total number of unique tokens in both strings.
    """
    tokens1 = set(nltk.word_tokenize(definition))
    tokens2 = set(nltk.word_tokenize(answer))
    overlap = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
    return overlap

def _compute_semantic_overlap(definition:str, answer:str):
    """
    Computes the semantic overlap between two vectors representing semantic features.
    
    Args:
        definition (str): A string representation of semantic features for the definition.
        answer (str): A string representation of semantic features for the answer.
    
    Returns:
        float: The semantic overlap score between the two vectors.
    """
    definition = np.array([float(x) for x in definition.split(" ")])
    answer = np.array([float(x) for x in answer.split(" ")])
    definition = definition.reshape(1, -1)
    answer = answer.reshape(1, -1)
    similarity_matrix = cosine_similarity(definition, answer)
    return similarity_matrix[0][0]

def compute_scores(disambiguated_data_path:str, subtask:str):
    """
    Computes and prints evaluation scores based on disambiguated data and gold data.
    
    Args:
        disambiguated_data_path (str): The path to the disambiguated data file.
    
    Returns:
        None
    """
    global instances

    gold_data = _get_gold_data(subtask)[0]
    disambiguated_data = _get_disambiguated_data(disambiguated_data_path)
    assert len(gold_data) == len(disambiguated_data)

    if args.pos == "ALL":
        number_of_evaluation_instances = len(gold_data)
    else:
        number_of_evaluation_instances = len([instance_gold for instance_gold in gold_data if instance_gold["pos"] == args.pos])

    true_labels = [1 for _ in range(number_of_evaluation_instances)]
    predicted_labels = [1 for _ in range(number_of_evaluation_instances)]

    correct, wrong = 0,0
    global_idx = 0

    for instance_gold, instance_disambiguated_data in zip(gold_data, disambiguated_data):
        if args.pos == "ALL":
            pass
        else:
            if instance_gold["pos"] != args.pos:
                continue
        assert instance_gold["id"] == instance_disambiguated_data["instance_id"]

        answer = instance_disambiguated_data["answer"]

        if args.subtask == "selection":
            if args.prompt_type == "v3":
                # adds [sense_key] before each gold definition
                for definition in instance_gold["definitions"]:
                    for idx_, gold_definition in enumerate(instance_gold["gold_definitions"]): # because there may be more than one gold candidate
                        if definition == gold_definition:
                            instance_gold["gold_definitions"][idx_] = f"[{instance_gold['gold'][idx_]}]: {instance_gold['gold_definitions'][idx_]}"
                # adds [sense_key] before all candidate definitions
                for idx, (sense_key, definition) in enumerate( list(zip(instance_gold["candidates"], instance_gold["definitions"])) ):
                    instance_gold["definitions"][idx] = f"[{sense_key}]: {definition}"
            else:
                # adds n) before each gold definition
                for idx, definition in enumerate(instance_gold["definitions"]):
                    for idx_, gold_definition in enumerate(instance_gold["gold_definitions"]): # because there may be more than one gold candidate
                        if definition == gold_definition:
                            instance_gold["gold_definitions"][idx_] = f"{idx+1}) {instance_gold['gold_definitions'][idx_]}"
                # adds n) before all candidate definitions
                for idx, definition in enumerate(instance_gold["definitions"]):
                    instance_gold["definitions"][idx] = f"{idx+1}) {definition}"

        if answer.strip() == "": selected_definition = ""
        else: selected_definition = _choose_definition(instance_gold, answer)
        
        if selected_definition in instance_gold["gold_definitions"]:
            correct += 1
        else:
            predicted_labels[global_idx] = 0
            wrong += 1

        global_idx += 1
    assert correct+wrong == number_of_evaluation_instances

    precision = precision_score(true_labels, predicted_labels, average='micro')
    recall = recall_score(true_labels, predicted_labels, average='micro')
    f1 = f1_score(true_labels, predicted_labels, average='micro')

    print("-----")
    print("Total number of instances:", number_of_evaluation_instances)
    print("Number of correctly classified instances:", correct)
    print("Number of incorrectly classified instances:", wrong)
    print()
    print("Precision (average=micro):", precision)
    print("Recall (average=micro):", recall)
    print("F1 Score (average=micro):", f1)
    print("Accuracy:", correct/number_of_evaluation_instances)

def _generate_gold_data_vectors(subtask:str):
    """
    Generates sentence embeddings for gold data and saves them to a file.

    Args:
        subtask (str)

    Returns:
        None
    """
    gold_vector_file_path = f"../data/evaluation/vectors/{args.sentence_embedder}_id2vec.tsv"
    if os.path.exists(gold_vector_file_path):
        print("Gold vectors already exist")
        return None
    gold_vector_folder_path = f"../data/evaluation/vectors"
    if not os.path.exists(gold_vector_folder_path):
        os.makedirs(gold_vector_folder_path)

    print("Generating vectors from gold data")
    sentence_embedder = SentenceTransformer(f'sentence-transformers/{args.sentence_embedder}')
    data = _get_gold_data(subtask)[0]

    gold_vector_file_path = f"../data/evaluation/vectors/{args.sentence_embedder}_id2vec.tsv"
    with open(gold_vector_file_path, "w") as fw:
        for el in tqdm(data, total=len(data)):
            id_ = el["id"]
            definitions = el["definitions"]
            for definition in definitions:
                vec = sentence_embedder.encode(definition).tolist()
                vec = " ".join([str(x) for x in vec])
                fw.write(f"{id_}\t{definition}\t{vec}\n")

def _generate_disambiguated_data_vectors(disambiguated_data_path:str, len_gold:int, is_finetuned:bool):
    """
    Generates sentence embeddings for disambiguated data and saves them to a file.

    Args:
        disambiguated_data_path (str): the path to the data disambiguated file 
        len_gold (int): number of instances in the gold file
        is_finetuned (bool): if the model is finetuned or not
    Returns:
        None
    """
    vector_file_path = f"../data/{args.subtask}/{args.prompt_type}/{args.prompt_addition}/{args.approach}/finetuned_{args.shortcut_model_name}/vectors/{args.sentence_embedder}_id2vec.tsv" if is_finetuned else f"../data/{args.subtask}/{args.prompt_type}/{args.prompt_addition}/{args.approach}/{args.shortcut_model_name}/vectors/{args.sentence_embedder}_id2vec.tsv"
    if os.path.exists(vector_file_path):
        with open(vector_file_path, "r") as fr:
            if len(fr.readlines()) != len_gold:
                print("Missing vectors")
                exit()
            print("Disambiguated data vectors already exist")
            return None
    vector_folder_path = f"../data/{args.subtask}/{args.prompt_type}/{args.prompt_addition}/{args.approach}/finetuned_{args.shortcut_model_name}/vectors" if is_finetuned else f"../data/{args.subtask}/{args.prompt_type}/{args.prompt_addition}/{args.approach}/{args.shortcut_model_name}/vectors"
    if not os.path.exists(vector_folder_path):
        os.makedirs(vector_folder_path)          

    print("Generating vectors from:", disambiguated_data_path)
    sentence_embedder = SentenceTransformer(f'sentence-transformers/{args.sentence_embedder}')

    with open(disambiguated_data_path) as fr:
        data = json.load(fr)

    vector_file_path = f"../data/{args.subtask}/{args.prompt_type}/{args.prompt_addition}/{args.approach}/finetuned_{args.shortcut_model_name}/vectors/{args.sentence_embedder}_id2vec.tsv" if is_finetuned else f"../data/{args.subtask}/{args.prompt_type}/{args.prompt_addition}/{args.approach}/{args.shortcut_model_name}/vectors/{args.sentence_embedder}_id2vec.tsv"
    id2vec_dd = {}
    with open(vector_file_path, "w") as fw:
        for el in tqdm(data, total=len(data)):
            id_ = el["instance_id"]
            answer = el["answer"]
            vec = sentence_embedder.encode(answer)
            id2vec_dd[id_] = " ".join([str(x) for x in vec.tolist()])
            vec = " ".join([str(x) for x in vec])
            fw.write(f"{id_}\t{vec}\n")

def _get_disambiguated_data(disambiguated_data_path:str):
    """
    Loads disambiguated data from a JSON file.

    Returns:
        dict: A dictionary containing the loaded gold data.
    """
    with open(disambiguated_data_path, "r") as json_file:
        disambiguated_data = json.load(json_file)
    return disambiguated_data

def _get_gold_data(subtask:str):
    """
    Loads gold data from a JSON file.

    Returns:
        dict: A dictionary containing the loaded gold data.
    """
    data = []
    if subtask in ["selection", "generation"]:
        data_path = "../data/evaluation/ALL_preprocessed.json"
        with open(data_path, "r") as json_file:
            data_ = json.load(json_file)
        data.append(data_)
    elif subtask == "wic":
        data_path = "../data/evaluation/test.en-en.data"
        gold_path = "../data/evaluation/test.en-en.gold"
        with open(data_path, "r") as fr:
            data_ = json.load(fr)
        with open(gold_path, "r") as fr:
            gold = json.load(fr)
        data.extend((data_, gold))
    return data

def _get_disambiguated_data_vectors(is_finetuned:bool):
    """
    Retrieves the disambiguated data sentence embeddings from a file and returns them as a dictionary.

    Args:
        is_finetuned (bool): if the model is finetuned or not

    Returns:
        dict: A dictionary mapping instance IDs to their corresponding sentence embeddings.
    """
    id2vec_disambiguated_data = {}

    vector_file_path = f"../data/{args.subtask}/{args.prompt_type}/{args.prompt_addition}/{args.approach}/finetuned_{args.shortcut_model_name}/vectors/{args.sentence_embedder}_id2vec.tsv" if is_finetuned else f"../data/{args.subtask}/{args.prompt_type}/{args.prompt_addition}/{args.approach}/{args.shortcut_model_name}/vectors/{args.sentence_embedder}_id2vec.tsv"
    with open(vector_file_path, "r") as fr:
        for line in fr:
            id_, vec = line.strip().split('\t')
            id2vec_disambiguated_data[id_] = vec
    return id2vec_disambiguated_data
  
def _get_gold_data_vectors():
    """
    Retrieves gold data sentence embeddings from a file and returns them as a dictionary.

    Args:
        None

    Returns:
        dict: A dictionary mapping instance IDs to their corresponding sentence embeddings.
    """
    id2vec_gold = {}

    with open(f"../data/evaluation/vectors/{args.sentence_embedder}_id2vec.tsv") as fr:
        for line in fr:
            id, definition, vec = line.strip().split('\t')
            id2vec_gold[id+definition] = vec
    return id2vec_gold

def compute_scores_wic(disambiguated_data_file:str, subtask:str):

    data_file, gold_file = _get_gold_data(subtask)
    answers = _get_disambiguated_data(disambiguated_data_file) 
    assert len(gold_file) == len(answers)

    if args.pos == "ALL":
        number_of_evaluation_instances = len(answers)
        ids_ = [instance_gold["id"] for instance_gold in data_file]
    else:
        number_of_evaluation_instances = len([instance_gold for instance_gold in data_file if instance_gold["pos"] == args.pos])
        ids_ = [instance_gold["id"] for instance_gold in data_file if instance_gold["pos"] == args.pos]

    true_labels = [1 for _ in range(number_of_evaluation_instances)]
    predicted_labels = [1 for _ in range(number_of_evaluation_instances)]

    correct, wrong = 0,0
    global_idx = 0

    for instance, instance_gold, instance_gpt in zip(data_file, gold_file, answers):

        if instance["id"] not in ids_:
            continue
        assert instance["id"] == instance_gold["id"] == instance_gpt["instance_id"]

        answer = instance_gpt["answer"]
        if "True" in answer and "False" in answer:
            predicted_labels[global_idx] = 0
            wrong += 1
            global_idx += 1
            continue
        elif "True" in answer and "False" not in answer:
            answer = "T"
        elif "False" in answer and "True" not in answer:
            answer = "F"

        if answer != instance_gold["tag"]:
            predicted_labels[global_idx] = 0
            wrong += 1
        else:
            correct+=1
        global_idx += 1
    assert correct+wrong == number_of_evaluation_instances

    precision = precision_score(true_labels, predicted_labels, average='micro')
    recall = recall_score(true_labels, predicted_labels, average='micro')
    f1 = f1_score(true_labels, predicted_labels, average='micro')

    print("-----")
    print("Total number of instances:", number_of_evaluation_instances)
    print("Number of correctly classified instances:", correct)
    print("Number of incorrectly classified instances:", wrong)
    print()
    print("Precision (average=micro):", precision)
    print("Recall (average=micro):", recall)
    print("F1 Score (average=micro):", f1)
    print("Accuracy:", correct/number_of_evaluation_instances)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script description')
    parser.add_argument('--subtask', '-s', type=str, help='Input the task')
    parser.add_argument("--prompt_type", "-pt", type=str, help="Input the prompt type")
    parser.add_argument("--prompt_addition", "-pa", type=str, default="no_additions", help="Input the prompt addition")
    parser.add_argument('--approach', '-a', type=str, help='Input the approach')
    parser.add_argument('--shortcut_model_name', '-sm', type=str, help='Input the model')
    parser.add_argument("--is_finetuned", "-f", type=bool, default=False, help="If the model we want to test is finetuned or not")
    parser.add_argument('--pos', '-p', type=str, help='Input the part of speech')
    parser.add_argument('--sentence_embedder', '-se', type=str, default=None, help='Input the sentence embedder')
    args = parser.parse_args()

    assert args.subtask in ["generation", "selection", "wic"]
    assert args.prompt_type in ["v1", "v1.1", "v1.2", "v1.3", "v2", "v2.1", "v2.2", "v3", "v3.1", "v3.2"]
    assert args.prompt_addition in ["no_additions", "cot", "reflective", "cognitive", "emotion"]
    assert args.approach in ["zero_shot", "one_shot", "few_shot"]
    supported_shortcut_model_names =  ["llama_2", "llama_3", "mistral", "falcon", "vicuna", 
                                      "tiny_llama", "stability_ai", "h2o_ai",
                                      "phi_3_small", "phi_3_mini", "gemma_2b", "gemma_9b"]
    assert args.shortcut_model_name in supported_shortcut_model_names
    
    assert args.pos in ["NOUN", "ADJ", "VERB", "ADV", "ALL"]

    disambiguated_data_path = f"../data/{args.subtask}/{args.prompt_type}/{args.prompt_addition}/{args.approach}/"
    if args.is_finetuned: disambiguated_data_path += f"finetuned_{args.shortcut_model_name}/output.json"
    else: disambiguated_data_path += f"{args.shortcut_model_name}/output.json"
    if args.subtask in ["selection", "generation"]:
        len_gold = len(_get_gold_data(args.subtask)[0])

        if args.subtask == "generation":
            assert args.sentence_embedder in ["all-mpnet-base-v2", "all-MiniLM-L6-v2", "Salesforce/SFR-Embedding-2_R"]
            _generate_gold_data_vectors(args.subtask)
            _generate_disambiguated_data_vectors(disambiguated_data_path, len_gold, args.is_finetuned)
            id2vec_gold = _get_gold_data_vectors()
            id2vec_disambiguated_data = _get_disambiguated_data_vectors(args.is_finetuned)
        compute_scores(disambiguated_data_path, args.subtask)

    elif args.subtask == "wic":
        compute_scores_wic(disambiguated_data_path, args.subtask)