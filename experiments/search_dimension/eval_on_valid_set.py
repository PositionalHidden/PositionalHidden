import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import sys
sys.path.append("../")
from positional_hidden.use_model import get_model_class


def format_chat_prompt(input,model_path,tokenizer):
    model_path=model_path.lower()

    if "mpt-7b-8k-instruct" in model_path:
        def format_prompt(instruction):
            template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n###Instruction\n{instruction}\n\n### Response\n"
            return template.format(instruction=instruction)
        prompt = format_prompt(input)

    elif "longchat" in model_path or "vicuna" in model_path:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], input)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    else:
        messages=[{"role":"user","content":input}]
        chat_template = tokenizer.chat_template
        if chat_template is None:
            raise ValueError("not support chat_template")

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt

def eval_dims(model,tokenizer,dataset,dim_to_scale):
    #set the dimensions to scale
    model.config.hidden_scale_config["dims"] = dim_to_scale
    losses = []
    device=model.device
    for i in tqdm(range(len(dataset)),desc="get validation loss when scaling {}".format(str(dim_to_scale))):
        prompt = dataset.iloc[i]["input"]
        prompt = format_chat_prompt(prompt, model.config._name_or_path, tokenizer)
        answer = dataset.iloc[i]["output"]

        prompt_ids = tokenizer.encode(prompt, return_tensors=None, padding=False, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, return_tensors=None, padding=False, add_special_tokens=False)
        input_ids = prompt_ids + answer_ids
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
        # labels
        labels = [-100] * len(prompt_ids) + answer_ids
        labels = torch.tensor(labels).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)

        #the tokens in the answer_ids need to be recomputed
        model.config.hidden_scale_config["last_recompute_tokens"] = len(answer_ids) + 1

        with torch.no_grad():
            with torch.autocast("cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                losses.append(loss.item())

    average_loss = sum(losses) / len(losses)
    del losses
    del model
    return round(average_loss, 4)

def eval_dims_main(model_path,dataset_path,dim_cand,topk=3)->list[int]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    AutoModelHiddenScale=get_model_class(model_path)
    model = AutoModelHiddenScale.from_pretrained(model_path,
                                             trust_remote_code=True,
                                             device_map="auto",
                                             torch_dtype="auto",
                                             attn_implementation="flash_attention_2",).eval()

    #configurations of the hidden states scaling
    model.config.hidden_scale_config = {
        "layers": range(10,26), # the layers to apply the scaling
        "dims": [], # the dimensions to apply the scaling
        "factor": 0, # the scaling factor
        "skip_first": 0, # skip the first n tokens when scaling hidden states
        "last_recompute_tokens": 1, # the number of tokens whose attention weights are recomputed
        "change_value": False, # whether to change the value states. If False, only the query and key states are modified
    }

    # load the validation set
    dataset = pd.read_json(dataset_path, lines=True)

    dim_losses={}

    #add baseline
    dim_losses["baseline"]=eval_dims(model,tokenizer,dataset,[])
    # for each dimension, scale it and evaluate the model's loss on the validation set
    for dim in dim_cand:
        loss=eval_dims(model,tokenizer,dataset,[dim])
        dim_losses[dim]=loss


    #sort by loss, and chose the topk dimensions
    dim_losses=sorted(dim_losses.items(), key=lambda x: x[1])
    dim_losses_topk=dict(dim_losses[:topk])
    print("dim and loss",dict(dim_losses))
    print("top-{} dim and loss".format(topk),dim_losses)

    return list(dim_losses_topk.keys())


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_path="meta-llama/Llama-2-7b-chat-hf"
    dataset_path= "valid_set/KV60_valid_set.json"
    dim_cand=[2158,2393]
    eval_dims_main(model_path,dataset_path,dim_cand,topk=3)
