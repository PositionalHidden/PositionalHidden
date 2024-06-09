import os
import dataclasses
import pathlib
import random
from copy import deepcopy
import argparse
import json
import logging
import statistics
import sys
import torch
from tqdm import tqdm
from xopen import xopen

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lost_in_the_middle.prompting import get_kv_retrieval_prompt

random.seed(0)
def get_model_class(model_path):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    architecture = config.architectures[0]

    from positional_hidden.models.modeling_llama_hidden_scale import LlamaForCausalLM, LlamaFlashAttention2, LlamaAttention, \
        LlamaSdpaAttention

    if "llama" in architecture.lower():
        Model = LlamaForCausalLM
    elif "mistral" in architecture.lower():
        from transformers.models.mistral.modeling_mistral import MistralForCausalLM, MistralFlashAttention2, \
            MistralAttention, MistralSdpaAttention
        MistralFlashAttention2.forward, MistralAttention.forward, MistralSdpaAttention.forward = LlamaFlashAttention2.forward, LlamaAttention.forward, LlamaSdpaAttention.forward
        MistralAttention._init_hidden_states_scale = LlamaAttention._init_hidden_states_scale
        Model = MistralForCausalLM
    elif "qwen2" in architecture.lower():
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2FlashAttention2, \
            Qwen2Attention, Qwen2SdpaAttention
        Qwen2FlashAttention2.forward, Qwen2Attention.forward, Qwen2SdpaAttention.forward = LlamaFlashAttention2.forward, LlamaAttention.forward, LlamaSdpaAttention.forward
        Qwen2Attention._init_hidden_states_scale = LlamaAttention._init_hidden_states_scale
        Model = Qwen2ForCausalLM
    elif "gemma" in architecture.lower():
        from positional_hidden.models.modeling_gemma_hidden_scale import GemmaForCausalLM
        Model = GemmaForCausalLM
    elif "mpt" in architecture.lower():
        from positional_hidden.models.modeling_mpt_hidden_scale import MptForCausalLM
        Model = MptForCausalLM
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return Model

def eval_kv(
    input_path,
    output_path,
):
    all_examples = []
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            all_examples.append(input_example)

    # Compute normal metrics in parallel, if applicable
    all_example_metrics = []
    for example in tqdm(all_examples):
        model_answer = example["model_answer"]

        accuracy = 1.0 if example["value"].lower() in model_answer.lower() else 0.0

        all_example_metrics.append(({"accuracy": accuracy}, example))

    # Average metrics across examples
    for metric_name in ["accuracy"]:
        average_metric_value = statistics.mean(
            example_metrics[metric_name] for (example_metrics, _) in all_example_metrics
        )
        print(f"{metric_name}: {average_metric_value}")

    if output_path:
        with xopen(output_path, "w") as f:
            for (example_metrics, example) in all_example_metrics:
                example_with_metrics = deepcopy(example)
                for metric_name, metric_value in example_metrics.items():
                    example_with_metrics[f"metric_{metric_name}"] = metric_value
                f.write(json.dumps(example_with_metrics) + "\n")

    return average_metric_value


def main(
    num_kvs,
    gold_list,
    model_path,
    scale_config_path,
    temperature,
    top_p,
    max_new_tokens,
    sample_num=None,
):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    #load scale config
    if scale_config_path is not None:
        with open(scale_config_path, "r") as f:
            scale_config = json.load(f)
        scale_config_name=os.path.basename(scale_config_path).replace(".json","")

    else:
        scale_config = None
        scale_config_name="no_scale"

    print("model name",os.path.basename(model_path))
    print("scale config:",scale_config)

    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)

    scores=[]
    for gold in gold_list:
        examples = []
        prompts = []
        all_model_ordered_kv_records = []

        input_path=script_dir+"/kv-retrieval-300_keys.jsonl.gz"
        # Fetch all of the prompts
        with xopen(input_path) as fin:
            for line in tqdm(fin):
                input_example = json.loads(line)
                # Get the prediction for the input example
                ordered_kv_records = deepcopy(input_example["ordered_kv_records"])
                key = input_example["key"]
                value = input_example["value"]

                original_kv_index = ordered_kv_records.index([key, value])
                # Remove the kv from its original index
                original_kv = ordered_kv_records.pop(original_kv_index)
                ordered_kv_records.insert(gold, original_kv)

                #select a subset of the kvs
                if num_kvs is not None:
                    ordered_kv_records = ordered_kv_records[:num_kvs]

                kv_prompt = get_kv_retrieval_prompt(
                    data=ordered_kv_records, key=key, query_aware_contextualization=False
                )

                kv_prompt = format_chat_prompt(kv_prompt, os.path.basename(model_path), tokenizer)
                prompts.append(kv_prompt)
                examples.append(deepcopy(input_example))
                all_model_ordered_kv_records.append(ordered_kv_records)

        #select a subset of the prompts
        prompts=prompts[:sample_num] if sample_num is not None else prompts

        # Get responses for all of the prompts
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

        config = AutoConfig.from_pretrained(model_path,trust_remote_code=True)


        #if don't apply scale dims
        if scale_config is None:
            Model=AutoModelForCausalLM
        else:
            Model = get_model_class(model_path)

        model = Model.from_pretrained(model_path,
                                      device_map="auto",
                                      trust_remote_code=True,
                                      torch_dtype="auto",
                                      attn_implementation="flash_attention_2" if not "mpt" in model_path.lower() else "eager",
                                      ).eval()

        model.config.hidden_scale_config = scale_config

        model_name=os.path.basename(model_path)
        test_dataset_name=os.path.basename(input_path).replace(".jsonl.gz","").replace("300",str(num_kvs))+"_gold_"+str(gold)+".jsonl"
        output_path=script_dir+"/model_responses/{model_name}/scale_{scale_config_name}/{test_dataset_name}".format(model_name=model_name,scale_config_name=scale_config_name,test_dataset_name=test_dataset_name)

        # Create directory for output path if it doesn't exist.
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        do_sample = temperature > 0.0

        responses = []
        for sample_index,prompt in tqdm(enumerate(prompts),desc="run on KV",total=len(prompts)):

            inputs=tokenizer([prompt], return_tensors="pt", padding=True,add_special_tokens=False).to(model.device)

            input_len=inputs['input_ids'].shape[1]

            with torch.no_grad():
                #with torch.autocast("cuda"):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p if do_sample else None,
                        use_cache=True,
                        return_dict_in_generate=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

            reply = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            responses.append(reply.strip())

        with xopen(output_path, "w") as f:
            for example, ordered_kv_records, prompt, response in zip(
                    examples, all_model_ordered_kv_records, prompts, responses
            ):
                output_example = deepcopy(example)
                # Add some extra metadata to the output example
                output_example["model_prompt"] = prompt
                output_example["model_answer"] = response
                output_example["model"] = model_name
                output_example["model_temperature"] = temperature
                output_example["model_top_p"] = top_p
                output_example["model_ordered_kv_records"] = ordered_kv_records
                f.write(json.dumps(output_example) + "\n")

        #get the score
        score=eval_kv(output_path, None)

        del model
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        scores.append({"gold":gold,"score":score,"num kvs":num_kvs})

    print(scores)
    #save
    with open(os.path.join(os.path.dirname(output_path),"scores.json"),"w") as f:
        json.dump(scores,f,indent=4)

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
            import warnings
            warnings.warn("chat_template is None, return the original input")
            return input

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-kvs", help="Number of kvs to use in the KV prompt", type=int, default=20)
    parser.add_argument("--gold", help="Gold doc index", type=int,nargs="+", default=[])

    parser.add_argument("--model-path", help="Model to use in generating responses", required=True)
    parser.add_argument("--scale-config", help="Configuration for scaling the hidden states", type=str, default=None)

    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=0.0)
    parser.add_argument("--top-p", help="Top-p to use in generation", type=float, default=1.0)
    parser.add_argument("--max-new-tokens",help="Maximum number of new tokens to generate",type=int,default=100)
    parser.add_argument("--sample-num", help="Number of samples to run", type=int, default=None)


    args = parser.parse_args()

    print("running on KV with the following arguments:")
    print("model path: ",args.model_path)
    print("scale config: ",args.scale_config)
    print("gold: ",args.gold)
    print("num kvs: ",args.num_kvs)
    print("temperature: ",args.temperature)
    print("top p: ",args.top_p)
    print("max new tokens: ",args.max_new_tokens)
    print("sample num: ",args.sample_num)

    main(
        num_kvs=args.num_kvs,
        gold_list=args.gold,
        model_path=args.model_path,
        scale_config_path=args.scale_config,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        sample_num=args.sample_num,
    )