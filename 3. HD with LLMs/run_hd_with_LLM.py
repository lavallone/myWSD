from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import warnings
import argparse
import time
import json
import torch
import os
import nltk
from sklearn.metrics import f1_score

SHORTCUT2FULLNAME = {"mistral" : "mistralai/Mistral-7B-Instruct-v0.2", "h2o_ai" : "h2oai/h2o-danube2-1.8b-chat", "llama_3" : "meta-llama/Meta-Llama-3-8B-Instruct", "gemma" : "google/gemma-2-9b-it"}

PROMPTS = {
           "wsd" : {
                    "zero_shot" : """Question: given the following sentence: "{text}", which one of the following definitions describes the semantics of the target word "{word}"?\nChoose the corresponding definition among: \n{candidate_definitions}.\nAnswer by reporting the corresponding definition and do not motivate your answer.\nAnswer:""",
                    "one_shot" : """Question: given the following sentence: "`` The sound of bells is a net to draw people into the church , `` he says .", which one of the following definitions describes the semantics of the target word "bells"?\nChoose the corresponding definition among: \n0) a hollow device made of metal that makes a ringing sound when struck.\n1) a push button at an outer door that gives a ringing or buzzing signal when pushed.\n2) the sound of a bell being struck.\n3) (nautical) each of the eight half-hour units of nautical time signaled by strokes of a ship\'s bell; eight bells signals 4:00, 8:00, or 12:00 o\'clock, either a.m. or p.m..\n4) the shape of a bell.\n5) a phonetician and father of Alexander Graham Bell (1819-1905).\n6) English painter; sister of Virginia Woolf; prominent member of the Bloomsbury Group (1879-1961).\n7) United States inventor (born in Scotland) of the telephone (1847-1922).\n8) a percussion instrument consisting of a set of tuned bells that are struck with a hammer; used as an orchestral instrument.\n9) the flared opening of a tubular device.\nAnswer by reporting the corresponding definition and do not motivate your answer.\nAnswer: 0) a hollow device made of metal that makes a ringing sound when struck.\n\nQuestion: given the following sentence: "{text}", which one of the following definitions describes the semantics of the target word "{word}"?\nChoose the corresponding definition among: \n{candidate_definitions}.\nAnswer by reporting the corresponding definition and do not motivate your answer.\nAnswer:""",
                    "few_shot" : """Question: given the following sentence: "The thing to lose sleep over is what people , having concluded that we are weaker than we are , are likely to do about it . The evidence suggests that foreign peoples believe the United States is weaker than the Soviet Union , and is bound to fall still further behind in the years ahead .", which one of the following definitions describes the semantics of the target word "bound"?\nChoose the corresponding definition among: \n0) confined by bonds.\n1) held with another element, substance or material in chemical or physical union.\n2) secured with a cover or binding; often used as a combining form.\n3) (usually followed by `to\') governed by fate.\n4) covered or wrapped with a bandage.\n5) headed or intending to head in a certain direction; often used as a combining form as in `college-bound students\'.\n6) bound by an oath.\n7) bound by contract.\n8) confined in the bowels.\nAnswer by reporting the corresponding definition and do not motivate your answer.\nAnswer: 3) (usually followed by `to\') governed by fate.\n\nQuestion: given the following sentence: "We will refer to the plane of the graph as the X-Y plane .", which one of the following definitions describes the semantics of the target word "plane"?\nChoose the corresponding definition among: \n0) an aircraft that has a fixed wing and is powered by propellers or jets.\n1) (mathematics) an unbounded two-dimensional shape.\n2) a level of existence or development.\n3) a power tool for smoothing or shaping wood.\n4) a carpenter\'s hand tool with an adjustable blade for smoothing or shaping wood.\nAnswer by reporting the corresponding definition and do not motivate your answer.\nAnswer: 1) (mathematics) an unbounded two-dimensional shape.\n\nQuestion: given the following sentence: "Do n\'t lie to your parents .", which one of the following definitions describes the semantics of the target word "lie"?\nChoose the corresponding definition among: \n0) be located or situated somewhere; occupy a certain position.\n1) be lying, be prostrate; be in a horizontal position.\n2) originate (in).\n3) be and remain in a particular state or condition.\n4) tell an untruth; pretend with intent to deceive.\n5) have a place in relation to something else.\n6) assume a reclining position.\nAnswer by reporting the corresponding definition and do not motivate your answer.\nAnswer: 4) tell an untruth; pretend with intent to deceive.\n\nQuestion: given the following sentence: "{text}", which one of the following definitions describes the semantics of the target word "{word}"?\nChoose the corresponding definition among: \n{candidate_definitions}.\nAnswer by reporting the corresponding definition and do not motivate your answer.\nAnswer:"""
                   },
           "hd" : {
                    "zero_shot" : """Question: given the following sentence: "{text}", which one of the following homonyms cluster describes the semantics of the target word "{word}"?\nEach homonym cluster is identified by a key and a set of definitions. Choose the corresponding homonym key among: \n{cluster_candidate_definitions}.\nAnswer by reporting the corresponding homonym key and do not motivate your answer.\nAnswer:""",
                    "one_shot" : """Question: given the following sentence: "`` The sound of bells is a net to draw people into the church , `` he says .", which one of the following homonyms cluster describes the semantics of the target word "bells"?\nEach homonym cluster is identified by a key and a set of definitions. Choose the corresponding homonym key among: \nbell.n.h.05 : ["a hollow device made of metal that makes a ringing sound when struck", "a push button at an outer door that gives a ringing or buzzing signal when pushed", "the sound of a bell being struck", "(nautical) each of the eight half-hour units of nautical time signaled by strokes of a ship\'s bell; eight bells signals 4:00, 8:00, or 12:00 o\'clock, either a.m. or p.m.", "the shape of a bell", "a percussion instrument consisting of a set of tuned bells that are struck with a hammer; used as an orchestral instrument", "the flared opening of a tubular device"].\nbell.n.h.04 : ["English painter; sister of Virginia Woolf; prominent member of the Bloomsbury Group (1879-1961)"].\nalexander_melville_bell.n.h.01 : ["a phonetician and father of Alexander Graham Bell (1819-1905)"].\nalexander_bell.n.h.01 : ["United States inventor (born in Scotland) of the telephone (1847-1922)"].\nAnswer by reporting the corresponding homonym key and do not motivate your answer.\nAnswer: bell.n.h.05.\n\nQuestion: given the following sentence: "{text}", which one of the following homonyms cluster describes the semantics of the target word "{word}"?\nEach homonym cluster is identified by a key and a set of definitions. Choose the corresponding homonym key among: \n{cluster_candidate_definitions}.\nAnswer by reporting the corresponding homonym key and do not motivate your answer.\nAnswer:""",
                    "few_shot" : """Question: given the following sentence: "The thing to lose sleep over is what people , having concluded that we are weaker than we are , are likely to do about it . The evidence suggests that foreign peoples believe the United States is weaker than the Soviet Union , and is bound to fall still further behind in the years ahead .", which one of the following homonyms cluster describes the semantics of the target word "bound"?\nEach homonym cluster is identified by a key and a set of definitions. Choose the corresponding homonym key among: \nbound.a.h.02 : ["confined by bonds", "held with another element, substance or material in chemical or physical union", "secured with a cover or binding; often used as a combining form", "(usually followed by `to\') governed by fate", "covered or wrapped with a bandage", "bound by an oath", "bound by contract", "confined in the bowels"].\nbound.a.h.01 : ["headed or intending to head in a certain direction; often used as a combining form as in `college-bound students\'"].\nAnswer by reporting the corresponding homonym key and do not motivate your answer.\nAnswer: bound.a.h.02.\n\nQuestion: given the following sentence: "We will refer to the plane of the graph as the X-Y plane .", which one of the following homonyms cluster describes the semantics of the target word "plane"?\nEach homonym cluster is identified by a key and a set of definitions. Choose the corresponding homonym key among: \nplane.n.h.01 : ["(mathematics) an unbounded two-dimensional shape", "a level of existence or development"].\nplane.n.h.03 : ["a power tool for smoothing or shaping wood", "a carpenter\'s hand tool with an adjustable blade for smoothing or shaping wood"].\naeroplane.n.h.01 : ["an aircraft that has a fixed wing and is powered by propellers or jets"].\nAnswer by reporting the corresponding homonym key and do not motivate your answer.\nAnswer: plane.n.h.01.\n\nQuestion: given the following sentence: "Do n\'t lie to your parents .", which one of the following homonyms cluster describes the semantics of the target word "lie"?\nEach homonym cluster is identified by a key and a set of definitions. Choose the corresponding homonym key among: \nlie.v.h.01 : ["be located or situated somewhere; occupy a certain position", "be lying, be prostrate; be in a horizontal position", "originate (in)", "be and remain in a particular state or condition", "have a place in relation to something else", "assume a reclining position"].\nlie.v.h.02 : ["tell an untruth; pretend with intent to deceive"].\nAnswer by reporting the corresponding homonym key and do not motivate your answer.\nAnswer: lie.v.h.02.\n\nQuestion: given the following sentence: "{text}", which one of the following homonyms cluster describes the semantics of the target word "{word}"?\nEach homonym cluster is identified by a key and a set of definitions. Choose the corresponding homonym key among: \n{cluster_candidate_definitions}.\nAnswer by reporting the corresponding homonym key and do not motivate your answer.\nAnswer:"""
                   },
          }

################
# DISAMBIGUATE #
################

def countdown(t):
    """
    Activates and displays a countdown timer in the terminal.

    Args:
        t (int): The duration of the countdown timer in seconds.

    Returns:
        None
    """
    while t > 0:
        mins, secs = divmod(t, 60)
        timer = '{:02d}'.format(secs)
        print(f"\033[1mWarning\033[0m: Found output files in the target directory! I will delete them in {timer}", end='\r')
        time.sleep(1)
        t -= 1

def _generate_prompt(instance:dict, eval_type:str, approach:str):
    
    word = instance["word"]
    text = instance["text"].replace(" ,", ",").replace(" .", ".")
    
    if eval_type == "wsd":
        candidate_definitions = "\n".join([f"{idx}) {x}" for idx, x in enumerate(instance["definitions"])])
        prompt = PROMPTS[eval_type][approach].format(
                word=word,
                text=text,
                candidate_definitions=candidate_definitions)
    else:
        cluster_candidate_definitions = "\n".join([f"{cluster_key} : {cluster_definitions_set}." for cluster_key, cluster_definitions_set in zip(instance["cluster_candidates"], instance["cluster_definitions"])])
        prompt = PROMPTS[eval_type][approach].format(
                word=word,
                text=text,
                cluster_candidate_definitions=cluster_candidate_definitions)
    
    return prompt

def _get_gold_data():
    """
    Loads gold data from a JSON file.

    Returns:
        dict: A dictionary containing the loaded gold data.
    """
    data_path = f"data/LLMantics/test.json"
    with open(data_path, "r") as json_file:
        gold_data = json.load(json_file)
    return gold_data

def _disambiguate_gpt(client, model, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[ {"role": "user", "content": prompt } ],
        max_tokens=25,
        )
    answer = response.choices[0].message.content
    return answer

def disambiguate(eval_type : str, approach : str, shortcut_model_name : str):

    gold_data = _get_gold_data()
    output_file_path = f"data/LLM_output/{eval_type}/{approach}/{shortcut_model_name}"
    n_instances_processed = 0
    json_data = []

    # to manage creation/deletion of folders
    if not os.path.exists(f"data/LLM_output/"):
        os.system(f"mkdir data/LLM_output/")
    if not os.path.exists(f"data/LLM_output/{eval_type}/"):
        os.system(f"mkdir data/LLM_output/{eval_type}/")
    if not os.path.exists(f"data/LLM_output/{eval_type}/{approach}/"):
        os.system(f"mkdir data/LLM_output/{eval_type}/{approach}/")
    if not os.path.exists(output_file_path):
        os.system(f"mkdir {output_file_path}")
    elif os.path.exists(f"{output_file_path}/output.txt"):
        countdown(5)
        os.system(f"rm -r {output_file_path}/*")

    full_model_name = SHORTCUT2FULLNAME[shortcut_model_name]
    tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(full_model_name, trust_remote_code=True, torch_dtype=torch.float16).cuda()
    pipe = pipeline("text-generation", model=model, device="cuda", tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, max_new_tokens=25)

    with open(f"{output_file_path}/output.txt", "a") as fa_txt, open(f"{output_file_path}/output.json", "w") as fw_json:
        for instance in tqdm(gold_data, total=len(gold_data)):

            n_instances_processed += 1
            instance_id = instance["id"]
            
            prompt = _generate_prompt(instance, eval_type, approach)
            #chat = [{"role": "user", "content": prompt}]
            #prompt_template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            answer = pipe(prompt)[0]["generated_text"].replace(prompt, "").replace("\n", "").strip()
            
            fa_txt.write(f"{instance_id}\t{answer}\n")
            fa_txt.flush()

            json_answer = {"instance_id":instance_id, "answer":answer}
            json_data.append(json_answer)
        json.dump(json_data, fw_json, indent=4)

#########
# SCORE #
#########

def _choose_definition(instance_gold, answer):
    definitions = instance_gold["definitions"]
    definition2overlap = {}
    for definition in definitions:
        overlap = _compute_lexical_overlap(definition, answer)
        definition2overlap[definition] = overlap
    return max(definition2overlap, key=definition2overlap.get)

def _choose_cluster_key(instance_gold, answer):
    tokenized_answer = list(nltk.word_tokenize(answer))
    for cluster_key in instance_gold["cluster_candidates"]:
        if cluster_key in tokenized_answer:
            return cluster_key
    # otherwise    
    cluster_candidates = instance_gold["cluster_candidates"]
    cluster_key2overlap = {}
    for cluster_candidate in cluster_candidates:
        overlap = _compute_lexical_overlap(cluster_candidate, answer)
        cluster_key2overlap[cluster_candidate] = overlap
    return max(cluster_key2overlap, key=cluster_key2overlap.get)

def _compute_lexical_overlap(definition, answer):
    tokens1 = set(nltk.word_tokenize(definition))
    tokens2 = set(nltk.word_tokenize(answer))
    overlap = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
    return overlap

def _get_disambiguated_data(disambiguated_data_path:str):
    with open(disambiguated_data_path, "r") as json_file:
        disambiguated_data = json.load(json_file)
    return disambiguated_data

def _filter_ids(gold_data, disambiguated_data, test_data):
    if test_data != "all":
        d = {"fga" : "test_FGA_ids.txt", "ha" : "test_HA_ids.txt", "ha_p" : "test_HA_p_ids.txt"}
        text_file = d[test_data]
        with open(f"data/LLMantics/subsets/{text_file}", "r") as file:
            instance_ids = [line.strip() for line in file]
        gold_data = [item for item in gold_data if item["id"] in instance_ids]
        disambiguated_data = [item for item in disambiguated_data if item["instance_id"] in instance_ids]
    return gold_data, disambiguated_data 

def score(test_data: str, eval_type : str, approach : str, shortcut_model_name : str):
    
    if eval_type == "wsd2hd": disambiguated_data_path = f"data/LLM_output/wsd/{approach}/{shortcut_model_name}/output.json"
    else: disambiguated_data_path = f"data/LLM_output/{eval_type}/{approach}/{shortcut_model_name}/output.json"
    gold_data = _get_gold_data()
    disambiguated_data = _get_disambiguated_data(disambiguated_data_path)
    gold_data, disambiguated_data = _filter_ids(gold_data, disambiguated_data, test_data)
    assert len(gold_data) == len(disambiguated_data)
    
    true_labels = [1 for _ in range(len(gold_data))]
    predicted_labels = [1 for _ in range(len(gold_data))]
    correct, wrong = 0,0
    global_idx = 0
    if eval_type == "wsd" or eval_type == "wsd2hd":
        for instance_gold, instance_disambiguated_data in zip(gold_data, disambiguated_data):
            assert instance_gold["id"] == instance_disambiguated_data["instance_id"]
            answer = instance_disambiguated_data["answer"]
            # adds n) before each gold definition
            for idx, definition in enumerate(instance_gold["definitions"]):
                for idx_, gold_definition in enumerate(instance_gold["gold_definitions"]): # because there may be more than one gold candidate
                    if definition == gold_definition:
                        instance_gold["gold_definitions"][idx_] = f"{idx}) {instance_gold['gold_definitions'][idx_]}"
            # adds n) before each cluster gold definition
            for idx, definition in enumerate(instance_gold["definitions"]):
                for idx_, cluster_gold_definition in enumerate(instance_gold["cluster_gold_definitions"][0]):
                    if definition == cluster_gold_definition:
                        instance_gold["cluster_gold_definitions"][0][idx_] = f"{idx}) {instance_gold['cluster_gold_definitions'][0][idx_]}"
            # adds n) before all candidate definitions
            for idx, definition in enumerate(instance_gold["definitions"]):
                instance_gold["definitions"][idx] = f"{idx}) {definition}"

            if answer.strip() == "":  wrong += 1; global_idx += 1; continue
            else:  selected_definition = _choose_definition(instance_gold, answer)
            
            if eval_type == "wsd2hd":
                if selected_definition in instance_gold["cluster_gold_definitions"][0]: correct += 1
                else: predicted_labels[global_idx] = 0; wrong += 1
            else: # "wsd"
                if selected_definition in instance_gold["gold_definitions"]: correct += 1
                else: predicted_labels[global_idx] = 0; wrong += 1
            global_idx += 1
    else: # "hd"    
        for instance_gold, instance_disambiguated_data in zip(gold_data, disambiguated_data):
            assert instance_gold["id"] == instance_disambiguated_data["instance_id"]
            answer = instance_disambiguated_data["answer"]
            if answer.strip() == "":  wrong += 1; global_idx += 1; continue
            else: selected_cluster_key = _choose_cluster_key(instance_gold, answer)
            
            if selected_cluster_key in instance_gold["cluster_gold"]: correct += 1
            else: predicted_labels[global_idx] = 0; wrong += 1
            global_idx += 1
        
    assert correct+wrong == len(gold_data)
    f1 = f1_score(true_labels, predicted_labels, average='micro')
    print("-----")
    print("Total number of instances:", len(gold_data))
    print("Number of correctly classified instances:", correct)
    print("Number of incorrectly classified instances:", wrong)
    print()
    print("F1 Score (average=micro):", f1)
        
if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    supported_mode = ["disambiguate", "score"]
    supported_test_data = ["all", "fga", "ha", "ha_p"]
    supported_eval_types = ["wsd", "wsd2hd", "hd"]
    supported_approaches = ["zero_shot", "one_shot", "few_shot"]
    supported_shortcut_model_names = ["mistral", "h2o_ai", "gpt_4", "llama_3", "gemma"]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str)
    parser.add_argument("--test_data", "-d", type=str, default="all")
    parser.add_argument("--eval_type", "-em", type=str)
    parser.add_argument("--approach", "-a", type=str)
    parser.add_argument("--shortcut_model_name", "-mn", type=str)
    args = parser.parse_args()
    
    assert args.test_data in supported_test_data
    assert args.eval_type in supported_eval_types
    assert args.approach in supported_approaches
    assert args.shortcut_model_name in supported_shortcut_model_names
    
    # when we disambiguate eval_type can only be "wsd" or "hd"
    assert args.mode!="disambiguate" or args.eval_type=="wsd" or args.eval_type=="hd"
    # and we only disambigaute the entire test dataset
    assert args.mode!="disambiguate" or args.test_data=="all"
    
    # this is regarding the evaluation (scoring) phase
    assert args.test_data!="fga" or args.eval_type=="wsd"
    assert args.test_data!="ha" or args.eval_type=="hd" or args.eval_type=="wsd2hd"
    assert args.test_data!="ha_p" or args.eval_type=="hd" or args.eval_type=="wsd2hd"
    
    if args.mode == "disambiguate":
        disambiguate(args.eval_type, args.approach, args.shortcut_model_name)
    else:
        score(args.test_data, args.eval_type, args.approach, args.shortcut_model_name)