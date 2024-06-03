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

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")

from lost_in_the_middle.prompting import (
    Document,
    get_qa_prompt,
)
from lost_in_the_middle.metrics import best_subspan_em

random.seed(0)

METRICS = [
    (best_subspan_em, "best_subspan_em"),
]

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

def eval_qa(
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
        all_example_metrics.append(get_metrics_for_example(example))

    # Average metrics across examples
    for (_, metric_name) in METRICS:
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


def get_metrics_for_example(example):
    gold_answers = example["answers"]
    model_answer = example["model_answer"]

    # NOTE: we take everything up to the first newline, since otherwise models could hack
    # the metric by simply copying te input context (as the gold answer is guaranteed
    # to occur in the input context).
    model_answer = model_answer.split("\n")[0].strip()

    example_metrics = {}
    for (metric, metric_name) in METRICS:
        example_metrics[metric_name] = metric(prediction=model_answer, ground_truths=gold_answers)
    return (example_metrics, example)


def main(
    num_docs,
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
        all_model_documents = []

        input_path=script_dir+"/qa_data/{num_docs}_total_documents/nq-open-{num_docs}_total_documents_gold_at_{gold}.jsonl.gz".format(num_docs=num_docs,gold=gold)

        # Fetch all of the prompts
        with xopen(input_path) as fin:
            for line in tqdm(fin):
                input_example = json.loads(line)
                # Get the prediction for the input example
                question = input_example["question"]

                documents = []
                for ctx in deepcopy(input_example["ctxs"]):
                    documents.append(Document.from_dict(ctx))
                if not documents:
                    raise ValueError(f"Did not find any documents for example: {input_example}")


                prompt = get_qa_prompt(
                    question,
                    documents,
                    mention_random_ordering=False,
                    query_aware_contextualization=False,
                )

                #apply chat template
                prompt = format_chat_prompt(prompt, os.path.basename(model_path), tokenizer)

                prompts.append(prompt)
                examples.append(deepcopy(input_example))
                all_model_documents.append(documents)

        #select a subset of the prompts
        prompts=prompts[:sample_num] if sample_num is not None else prompts

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

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
        test_dataset_name=os.path.basename(input_path).replace(".jsonl.gz","jsonl")
        output_path=script_dir+"/model_responses/{model_name}/scale_{scale_config_name}/{test_dataset_name}".format(model_name=model_name,scale_config_name=scale_config_name,test_dataset_name=test_dataset_name)

        # Create directory for output path if it doesn't exist.
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        do_sample = temperature > 0.0

        responses = []
        for sample_index,prompt in tqdm(enumerate(prompts),desc="run on NQ",total=len(prompts)):

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
            for example, model_documents, prompt, response in zip(examples, all_model_documents, prompts, responses):
                output_example = deepcopy(example)
                # Add some extra metadata to the output example
                output_example["model_prompt"] = prompt
                output_example["model_documents"] = [dataclasses.asdict(document) for document in model_documents]
                output_example["model_answer"] = response
                output_example["model"] = model_path
                output_example["model_temperature"] = temperature
                output_example["model_top_p"] = top_p
                f.write(json.dumps(output_example) + "\n")

        #get the score
        score=eval_qa(output_path, None)

        del model
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        scores.append({"gold":gold,"score":score,"num docs":num_docs})

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
    parser.add_argument("--num-docs", help="Number of documents to use in the QA prompt", type=int, default=20)
    parser.add_argument("--gold", help="Gold doc index", type=int,nargs="+", default=[])

    parser.add_argument("--model-path", help="Model to use in generating responses", required=True)
    parser.add_argument("--scale-config", help="Configuration for scaling the hidden states", type=str, default=None)

    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=0.0)
    parser.add_argument("--top-p", help="Top-p to use in generation", type=float, default=1.0)
    parser.add_argument("--max-new-tokens",help="Maximum number of new tokens to generate",type=int,default=100)
    parser.add_argument("--sample-num", help="Number of samples to run", type=int, default=None)


    args = parser.parse_args()

    main(
        num_docs=args.num_docs,
        gold_list=args.gold,
        model_path=args.model_path,
        scale_config_path=args.scale_config,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        sample_num=args.sample_num,
    )