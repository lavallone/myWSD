import torch
# nltk
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')

# here we can implement different EVAL pipelines that come to our minds

def eval_pipeline(eval_input_list):
    ris = []
    for i in range(len(eval_input_list)):
        answer_tokenized = nltk.word_tokenize(eval_input_list[i]["answer"].lower())
        gold_definition = " ".join(eval_input_list[i]["gold_definitions"])
        gold_definition_tokenized = nltk.word_tokenize(gold_definition.lower())
        # preprocessing
        stop_words = set(stopwords.words('english'))
        answer_tokenized_set = set([t for t in answer_tokenized if t.isalnum() and t not in stop_words])
        gold_definition_tokenized_set = set([t for t in gold_definition_tokenized if t.isalnum() and t not in stop_words])
        # IoU
        intersection = answer_tokenized_set.intersection(gold_definition_tokenized_set)
        union = answer_tokenized_set.union(gold_definition_tokenized_set)
        lexical_overlap = len(intersection) / len(union)
        ris.append(lexical_overlap)
    return torch.tensor(ris).mean()