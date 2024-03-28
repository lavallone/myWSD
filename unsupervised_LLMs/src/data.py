import json
from tqdm import tqdm
from torch.utils.data import Dataset

# dataset class
class WSD_Dataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.make_data()
        
    # utility function for generating PROMPT templates
    def generate_prompt(self, text, word, definitions):
        prompt = "QUESTION: select the most suitable meaning for \""
        prompt += word + "\" in the following sentence: " + text +"\n"
        prompt += "Choose the corresponding definition among: "
        for i in range(len(definitions)):
            prompt += "\n" + str(i) + ") " + definitions[i]
        prompt += "\nAnswer by reporting the corresponding definition and do not motivate your answer.\nANSWER: "
        return prompt

    # "PytorchLightning fashion"
    def make_data(self):
        ris = []
        with open(self.data_path) as f:
            data = json.load(f)
        for i in tqdm(range(len(data))):
            item = {}
            item["id"] = data[i]["id"]
            item["prompt"] = self.generate_prompt(data[i]["text"], data[i]["word"], data[i]["definitions"])
            item["gold_definitions"] = data[i]["gold_definitions"]
            ris.append(item)
        return ris
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @staticmethod
    def collate_batch(batch):
        batch_out = dict()
        batch_out["ids"] = [sample["id"] for sample in batch]
        batch_out["prompts"] = [sample["prompt"] for sample in batch]
        batch_out["gold_definitions"] = [sample["gold_definitions"] for sample in batch]
        return batch_out
    
def data_post_processing(model_name, outputs):
    if model_name[:4] == "meta" or model_name[:7] == "mistral":
        new_outputs = [ out[out.find("ANSWER")+11:] for out in outputs ]
    elif model_name[7:13] == "falcon":
        new_outputs = [ out[out.find("ANSWER"):] for out in outputs ]
    else:
        print("model not included in the post-processing phase")
    if model_name[:7] == "mistral" or model_name[7:13] == "falcon":  # in mistral we need to do this additional operations
        new_outputs = [ e[:e.find("\n")] for e in new_outputs ]
    return new_outputs