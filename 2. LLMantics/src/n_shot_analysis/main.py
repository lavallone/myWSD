from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from variables import shortcut_model_name2full_model_name, prompts, chat_template_prompts
from tqdm import tqdm
import warnings
import argparse
import time
import json
import torch
import os
import nltk
import numpy as np

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
        _, secs = divmod(t, 60)
        timer = '{:02d}'.format(secs)
        print(f"\033[1mWarning\033[0m: Found output files in the target directory! I will delete them in {timer}", end='\r')
        time.sleep(1)
        t -= 1

def _generate_prompt(instance:dict, analysis_type:str, ambiguity:str, most_frequent:str, approach:str, chat_template = False):

    word = instance["word"]
    text = instance["text"].replace(" ,", ",").replace(" .", ".")
    candidate_definitions = "\n".join([f"{idx+1}) {x}" for idx, x in enumerate(instance["definitions"])])
    
    # if we use chat_template, we need to deal with each approach differently
    if chat_template:
        chat_template_prompt_dict = chat_template_prompts[analysis_type][ambiguity][most_frequent][approach]
        if approach == "one_shot": 
            prompt = [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}, {"role": "user", "content": ""}]
            prompt[0]["content"] = chat_template_prompt_dict["example"]
            prompt[1]["content"] = chat_template_prompt_dict["example_output"]
            prompt[2]["content"] = chat_template_prompt_dict["prompt"].format(word=word,
                                                                                text=text,
                                                                                candidate_definitions=candidate_definitions)
        elif approach == "few_shot": 
            prompt = [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}, {"role": "user", "content": ""}]
            prompt[0]["content"] = chat_template_prompt_dict["example_1"]
            prompt[1]["content"] = chat_template_prompt_dict["example_1_output"]
            prompt[2]["content"] = chat_template_prompt_dict["example_2"]
            prompt[3]["content"] = chat_template_prompt_dict["example_2_output"]
            prompt[4]["content"] = chat_template_prompt_dict["example_3"]
            prompt[5]["content"] = chat_template_prompt_dict["example_3_output"]
            prompt[6]["content"] = chat_template_prompt_dict["prompt"].format(word=word,
                                                                                text=text,
                                                                                candidate_definitions=candidate_definitions)
    # otherwise we create the simple prompt
    else:
        prompt = prompts[analysis_type][ambiguity][most_frequent][approach].format(
                word=word,
                text=text,
                candidate_definitions=candidate_definitions)
    return prompt

def _get_gold_data():
    """
    Loads gold data from a JSON file.

    Returns:
        dict: A dictionary containing the loaded gold data.
    """
    data_path = "../../data/evaluation/ALL_preprocessed.json"
    with open(data_path, "r") as json_file:
        gold_data = json.load(json_file)
    return gold_data

def disambiguate(analysis_type:str, ambiguity:str, most_frequent:str, approach:str, shortcut_model_name:str):
    
    assert analysis_type in supported_analysis_type
    assert shortcut_model_name in supported_shortcut_model_names
    assert ambiguity in supported_ambiguity
    assert approach in supported_approaches
    assert most_frequent in supported_mfs
    global shortcut_model_name2full_model_name

    gold_data = _get_gold_data()
    output_file_path = f"../../data/n_shot_analysis/{analysis_type}/{ambiguity}/{most_frequent}/{approach}/{shortcut_model_name}"
    n_instances_processed = 0
    json_data = []

    # to manage creation/deletion of folders
    if not os.path.exists(f"../../data/n_shot_analysis/"):
        os.system(f"mkdir ../../data/n_shot_analysis/")
    if not os.path.exists(f"../../data/n_shot_analysis/{analysis_type}/"):
        os.system(f"mkdir ../../data/n_shot_analysis/{analysis_type}/")
    if not os.path.exists(f"../../data/n_shot_analysis/{analysis_type}/{ambiguity}/"):
        os.system(f"mkdir ../../data/n_shot_analysis/{analysis_type}/{ambiguity}/")
    if not os.path.exists(f"../../data/n_shot_analysis/{analysis_type}/{ambiguity}/{most_frequent}/"):
        os.system(f"mkdir ../../data/n_shot_analysis/{analysis_type}/{ambiguity}/{most_frequent}/")
    if not os.path.exists(f"../../data/n_shot_analysis/{analysis_type}/{ambiguity}/{most_frequent}/{approach}/"):
        os.system(f"mkdir ../../data/n_shot_analysis/{analysis_type}/{ambiguity}/{most_frequent}/{approach}/")
    if not os.path.exists(output_file_path):
        os.system(f"mkdir {output_file_path}")
    elif os.path.exists(f"{output_file_path}/output.txt"):
        countdown(5)
        os.system(f"rm -r {output_file_path}/*")
        
    # only for the falcon model we set trust_remote_code to False
    trust_remote_code = True
    if shortcut_model_name == "falcon": trust_remote_code = False

    full_model_name = shortcut_model_name2full_model_name[shortcut_model_name]
    tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    if shortcut_model_name == "phi_3_mini": model = AutoModelForCausalLM.from_pretrained(full_model_name, trust_remote_code=trust_remote_code, torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda()
    else: model = AutoModelForCausalLM.from_pretrained(full_model_name, trust_remote_code=trust_remote_code, torch_dtype=torch.float16).cuda()
    pipe = pipeline("text-generation", model=model, device="cuda", tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, max_new_tokens=25)

    with open(f"{output_file_path}/output.txt", "a") as fa_txt, open(f"{output_file_path}/output.json", "w") as fw_json:
        for instance in tqdm(gold_data, total=len(gold_data)):

            n_instances_processed += 1
            instance_id = instance["id"]
            
            # only these two models doesn't support chat_template feature
            if shortcut_model_name == "vicuna" or shortcut_model_name == "falcon":
                prompt = _generate_prompt(instance, analysis_type, ambiguity, most_frequent, approach)
                answer = pipe(prompt)[0]["generated_text"].replace(prompt, "").replace("\n", "").strip()
            else:
                chat_prompt = _generate_prompt(instance, analysis_type, ambiguity, most_frequent, approach, chat_template = True)
                prompt_template = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)
                answer = pipe(prompt_template)[0]["generated_text"].replace(prompt_template, "").replace("\n", "").strip()
            
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

def _compute_lexical_overlap(definition, answer):
    tokens1 = set(nltk.word_tokenize(definition))
    tokens2 = set(nltk.word_tokenize(answer))
    overlap = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
    return overlap

def _get_disambiguated_data(disambiguated_data_path:str):
    with open(disambiguated_data_path, "r") as json_file:
        disambiguated_data = json.load(json_file)
    return disambiguated_data

def compute_scores(disambiguated_data_path:str):

    gold_data = _get_gold_data()
    disambiguated_data = _get_disambiguated_data(disambiguated_data_path)
    assert len(gold_data) == len(disambiguated_data)

    correct_most_frequent, not_correct_most_frequent, correct_not_most_frequent, not_correct_not_most_frequent = 0, 0, 0, 0
    correct, wrong = 0,0
    global_idx = 0

    for instance_gold, instance_disambiguated_data in zip(gold_data, disambiguated_data):
        assert instance_gold["id"] == instance_disambiguated_data["instance_id"]

        answer = instance_disambiguated_data["answer"]

        # adds n) before each gold definition
        for idx, definition in enumerate(instance_gold["definitions"]):
            for idx_, gold_definition in enumerate(instance_gold["gold_definitions"]): # because there may be more than one gold candidate
                if definition == gold_definition:
                    instance_gold["gold_definitions"][idx_] = f"{idx}+1) {instance_gold['gold_definitions'][idx_]}"
        # adds n) before all candidate definitions
        for idx, definition in enumerate(instance_gold["definitions"]):
            instance_gold["definitions"][idx] = f"{idx}+1) {definition}"
        
        # we need it for differentiate mfs and not_mfs answers
        def2idx = {d:i for i,d in enumerate(instance_gold["definitions"])}

        # if model answers "", we consider as not_mfs and wrong prediction
        if answer.strip() == "": 
            not_correct_not_most_frequent+=1
            wrong += 1
            global_idx += 1
            continue
        else: selected_definition = _choose_definition(instance_gold, answer)
        
        if selected_definition in instance_gold["gold_definitions"]:
            if def2idx[selected_definition]==0: correct_most_frequent+=1
            else: correct_not_most_frequent+=1
            correct += 1
        else:
            if def2idx[selected_definition]==0: not_correct_most_frequent+=1
            else: not_correct_not_most_frequent+=1
            wrong += 1

        global_idx += 1
    assert correct+wrong == len(gold_data)
    
    perc_mfs_predicted = round(((correct_most_frequent+not_correct_most_frequent)/len(gold_data))*100,2)
    perc_not_mfs_correctly_predicted = round( ((correct_not_most_frequent)/(correct_not_most_frequent+not_correct_not_most_frequent))*100 , 2)
    acc = round((correct/len(gold_data))*100,2)
    return perc_mfs_predicted, perc_not_mfs_correctly_predicted, acc

def score(analysis_type:str, approach:str, shortcut_model_name:str):
    
    if analysis_type == "mfs_analysis":
        # MFS analysis
        ambiguity_level = "6_candidates"
        most_frequent_list = ["mfs", "not_mfs"]
        mfs_ris = []
        for most_frequent in most_frequent_list:
            disambiguated_data_path = f"../../data/n_shot_analysis/{analysis_type}/{ambiguity_level}/{most_frequent}/{approach}/{shortcut_model_name}/output.json"
            perc_mfs_predicted, perc_not_mfs_correctly_predicted, acc = compute_scores(disambiguated_data_path)
            mfs_ris.append([perc_mfs_predicted, perc_not_mfs_correctly_predicted, acc])
        print("# MFS analysis")
        table_values=[["", "% mfs predicted", "% not mfs predicted correctly", "f1-score"],
                    ["MFS", str(mfs_ris[0][0])+"%", str(mfs_ris[0][1])+"%", str(mfs_ris[0][2])],
                    ["not MFS", str(mfs_ris[1][0])+"%", str(mfs_ris[1][1])+"%", str(mfs_ris[1][2])]]
        col_widths = [max(len(str(cell)) for cell in column) for column in zip(*table_values)]
        for row in table_values:
            print(" | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths)))
        print(f"The loss is about -{mfs_ris[0][2]-mfs_ris[1][2]}.")
    
    else:
        # AMBIGUITY analysis
        ambiguity_list = ["1_candidate", "3_candidates", "6_candidates", "10_candidates", "16_candidates"]
        most_frequent_list = ["mfs", "not_mfs"]
        ambiguity_ris = []
        for most_frequent in most_frequent_list:
            l = []
            for ambiguity_level in tqdm(ambiguity_list, total=len(ambiguity_list)):
                if ambiguity_level == "1_candidate" and most_frequent == "not_mfs":
                    continue
                disambiguated_data_path = f"../../data/n_shot_analysis/{analysis_type}/{ambiguity_level}/{most_frequent}/{approach}/{shortcut_model_name}/output.json"
                _, _, acc = compute_scores(disambiguated_data_path)
                l.append(acc)
            std = np.asarray(l).std()
            l.append(std)
            ambiguity_ris.append(l)
        print("# AMBIGUITY analysis")
        table1_values=[["", "#1", "#3", "#6", "#10", "#16", "std"],
                    ["MFS", str(ambiguity_ris[0][0]), str(ambiguity_ris[0][1]), str(ambiguity_ris[0][2]), str(ambiguity_ris[0][3]), str(ambiguity_ris[0][4]), str(ambiguity_ris[0][5])],]
        col_widths = [max(len(str(cell)) for cell in column) for column in zip(*table1_values)]
        for row in table1_values:
            print(" | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths)))
        print()
        table2_values=[["", "#3", "#6", "#10", "#16", "std"],
                    ["not MFS", str(ambiguity_ris[1][0]), str(ambiguity_ris[1][1]), str(ambiguity_ris[1][2]), str(ambiguity_ris[1][3]), str(ambiguity_ris[1][4])]]
        col_widths = [max(len(str(cell)) for cell in column) for column in zip(*table2_values)]
        for row in table2_values:
            print(" | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths)))

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    supported_analysis_type = ["mfs_analysis", "ambiguity_analysis"]
    supported_mode = ["disambiguate", "score"]
    supported_ambiguity = ["1_candidate", "3_candidates", "6_candidates", "10_candidates", "16_candidates"]
    supported_mfs = ["mfs", "not_mfs"]
    supported_approaches = ["one_shot", "few_shot"]
    supported_shortcut_model_names = ["llama_2", "llama_3", "mistral", "falcon", "vicuna", 
                                      "tiny_llama", "stability_ai", "h2o_ai",
                                      "phi_3_small", "phi_3_mini", "gemma_2b", "gemma_9b"]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_type", "-at", type=str, help="The type of analysis we want to conduct")
    parser.add_argument("--mode", "-m", type=str, help="Input the mode (disambiguate or score)")
    parser.add_argument("--ambiguity", "-am", type=str, default="6_candidates", help="Input the ambiguity level")
    parser.add_argument("--most_frequent", "-mf", type=str, default="mfs", help="Input the most frequent level")
    parser.add_argument("--approach", "-a", type=str, help="Input the approach")
    parser.add_argument("--shortcut_model_name", "-mn", type=str, help="Input the model")
    args = parser.parse_args()
    
    assert args.ambiguity!="1_candidate" or args.most_frequent=="mfs" # ambiguity level 1 implies mfs
    assert args.analysis_type=="ambiguity_analysis" or args.ambiguity=="6_candidates" # doing mfs analyzing implies 6 candidates as ambiguity level
    
    if args.mode == "disambiguate":
        disambiguate(args.analysis_type, args.ambiguity, args.most_frequent, args.approach, args.shortcut_model_name)
    else:
        score(args.analysis_type, args.approach, args.shortcut_model_name)