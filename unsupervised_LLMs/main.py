from torch.utils.data import DataLoader
import json
from src.data import WSD_Dataset
from src.eval import eval_pipeline
from src.LLMs_generation import generate_with_pipeline

if __name__ == '__main__':
    model_name = "tiiuae/falcon-7b"
    dataset = WSD_Dataset("data/ALL_preprocessed.json")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=WSD_Dataset.collate_batch)
    to_quant = True # if we need to quantize the model or not
    
    eval_input_list = generate_with_pipeline(model_name, dataloader, to_quant)
    #eval_input_list = generate(model_name, dataloader)
    
    # we save generated data
    model_name = model_name.replace('/', '-')
    json.dump(eval_input_list, open("data/"+model_name+"_generated_data.json", "w"))
    # EVAL phase
    eval_ris = eval_pipeline(eval_input_list)
    print(eval_ris)