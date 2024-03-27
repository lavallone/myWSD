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