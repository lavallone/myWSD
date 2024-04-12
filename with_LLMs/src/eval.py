import torch
# nltk
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from torch.nn import CosineSimilarity
from sentence_transformers import SentenceTransformer

# EVAL pipelines
# - selection
# - generation

def eval_selection(eval_input_list):
    ris = []
    for i in range(len(eval_input_list)):
        answer_tokenized = nltk.word_tokenize(eval_input_list[i]["answer"].lower())
        gold_definition = " ".join(eval_input_list[i]["gold_definitions"])
        gold_definition_tokenized = nltk.word_tokenize(gold_definition.lower())
        # preprocessing
        answer_tokenized_set = set([t for t in answer_tokenized])
        gold_definition_tokenized_set = set([t for t in gold_definition_tokenized])
        # IoU
        intersection = answer_tokenized_set.intersection(gold_definition_tokenized_set)
        union = answer_tokenized_set.union(gold_definition_tokenized_set)
        lexical_overlap = len(intersection) / len(union)
        ris.append(lexical_overlap)
    return f"| SELECTION SCORE is {round(torch.tensor(ris).mean().item(), 4)} |"

def eval_generation(eval_input_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)
    ris = []
    for i in range(len(eval_input_list)):
        answer = eval_input_list[i]["answer"].lower()
        gold_definition = " ".join(eval_input_list[i]["gold_definitions"]).lower()
        emb_1 = model.encode(answer, convert_to_tensor=True).reshape(1, -1)
        emb_2 = model.encode(gold_definition, convert_to_tensor=True).reshape(1, -1)
        
        cos_sim = CosineSimilarity(dim=1, eps=1e-6)
        similarity_score = cos_sim(emb_1, emb_2).item()
        ris.append(similarity_score)
    return f"| GENERATION SCORE is {round(torch.tensor(ris).mean().item(), 4)} |"