import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from src.eval import eval_pipeline
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
def generate_with_pipeline(model_name, dataloader, to_quant=False):
    if to_quant:
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True) # when we use bigger models
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
    
    eval_input_list = []
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # with 'pipeline' we can more easily set generation parameters
        generated_sequences = generation_pipeline(
            batch["prompts"],
            max_new_tokens=50,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        outputs = [ out[0]["generated_text"] for out in generated_sequences ]
        new_outputs = data_post_processing(model_name, outputs)
        print(new_outputs[0])
        
        for i in range(len(new_outputs)):
            d = {"id" : batch["ids"][i], "answer" : new_outputs[i], "gold_definitions" : batch["gold_definitions"][i]}
            # filtering for evaluation
            if new_outputs[i] == "" or "QUESTION" in new_outputs[i]: continue
            eval_input_list.append(d)
        
        # each n iterations we print EVAL infos
        if step % 20 == 0 and step!=0:
            break
            print(eval_pipeline(eval_input_list))
        # after x steps we finish the inferring phase
        if step == 200: break
        
    return eval_input_list