import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from src.eval import eval_selection, eval_generation
from src.data import data_post_processing
## LLAMA2
# huggingface-cli login ---> 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

# def generate(model_name, dataloader):
#     tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
#     tokenizer.pad_token = tokenizer.eos_token
#     model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True).to("cuda")#, load_in_8bit=True, pad_token_id=0)
#     model.bfloat16() # it fixes a problem with Llama2
    
#     eval_input_list = []
#     for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
#         tokenized_prompts = tokenizer(batch["prompts"], return_tensors="pt", padding=True).to("cuda")
#         tokenized_outputs = model.generate(**tokenized_prompts, max_new_tokens=50)
#         outputs = tokenizer.batch_decode(tokenized_outputs, skip_special_tokens=True)
#         new_outputs = [ out[out.rfind("ANSWER")+11:] for out in outputs ]
#         for i in range(len(new_outputs)):
#             d = {"id" : batch["ids"][i], "answer" : new_outputs[i], "gold_definitions" : batch["gold_definitions"][i]}
#             eval_input_list.append(d)
#         # each n iterations we print EVAL infos
#         if step % 1 == 0:
#             print(eval_pipeline(eval_input_list))
#     return eval_input_list

# maybe a better choice!
def generate_with_pipeline(model_name, dataset, to_quant=False, eval_type="selection"):
    if to_quant:
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True) # when we use bigger models
        model = model.bfloat16()
    else:
        model = model_name # is simply a string representing the model name
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    generation_pipeline.tokenizer.pad_token_id = model.config.eos_token_id
    
    generated_sequences = generation_pipeline(KeyDataset(dataset, "prompt"), 
                                                batch_size=4, 
                                                max_new_tokens=25,
                                                num_return_sequences=1,
                                                eos_token_id=tokenizer.eos_token_id,)
    outputs = []
    for output in tqdm(generated_sequences):
        outputs += [ e["generated_text"] for e in output]
    prompts = [ dataset.data[i]["prompt"] for i in range(len(dataset.data)) ]
    assert len(outputs) == len(prompts)
    new_outputs = data_post_processing(prompts, outputs)
    
    eval_input_list = [ {"id" : dataset.data[i]["id"], "answer" : new_outputs[i], "gold_definitions" : dataset.data[i]["gold_definitions"]} for i in range(len(dataset.data)) ]
    return eval_input_list