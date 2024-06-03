# Encoding: UTF-8
import os
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")

import pandas as pd
import pathlib

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
import torch

from transformers import AutoTokenizer,AutoModelForCausalLM

from tqdm import tqdm
from eval import eval_all
import json

DATASET2PROMPT = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    #"passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nLet's think step by step, and then give the number of the paragraph that the abstract is from.",
    #"passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nLet's think step by step. You must first analyse the abstract and the relevant paragraph, and then give the number of the paragraph that the abstract is from.",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}

DATASET2MAXLEN ={
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}

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

#根据文件名决定prompt函数
def prompt_func(df_path,input,context):
    df_name=os.path.basename(df_path).replace('.jsonl','')
    if df_name.endswith('_e'):
        df_name=df_name[:-2]
    elif df_name.endswith('_e_sampled'):
        df_name = df_name[:-10]
    prompt=DATASET2PROMPT[df_name]
    prompt=prompt.format(input=input,context=context)
    return prompt


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
            raise ValueError("模型不支持chat_template")

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


    return prompt



def run_on_dataset(df_path, model,tokenizer, scale_config_name, max_length):

    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    print('test on :',df_path)
    df_name=os.path.basename(df_path)
    test_dataset = pd.read_json(df_path, lines=True)
    test_dataset['model_reply'] = ''

    model_path = model.config._name_or_path
    output_path = script_dir+"/model_responses/{model_name}/scale_{scale_config_name}/{test_dataset_name}".format(
        model_name=os.path.basename(model_path), scale_config_name=scale_config_name, test_dataset_name=df_name)
    pathlib.Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    #get max_new_tokens
    df_name_original=os.path.basename(df_path).replace('.jsonl','')
    if df_name_original.endswith('_e'):
        df_name_original=df_name_original[:-2]
    elif df_name_original.endswith('_e_sampled'):
        df_name_original = df_name_original[:-10]
    max_new_tokens=DATASET2MAXLEN[df_name_original]

    for i in tqdm(range(len(test_dataset)),desc='eval on longbench',total=len(test_dataset),mininterval=1):
        torch.cuda.empty_cache()

        context=test_dataset.iloc[i]['context']
        input=test_dataset.iloc[i]['input']
        prompt=prompt_func(df_path,input,context)

        # if length of tokenized_prompt is longer than max_length, truncate and rejoin
        tokenized_prompt=tokenizer(prompt, return_tensors="pt", padding=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                tokenized_prompt[-half:], skip_special_tokens=True)


        # fewshot type tasks do not need chat format
        if df_name_original not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = format_chat_prompt(prompt, model_path, tokenizer)

        inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
        context_length = inputs.input_ids.shape[-1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=1,
            use_cache=True,
            return_dict_in_generate=False,
            pad_token_id=tokenizer.eos_token_id,
            min_new_tokens=1,
        )

        reply = tokenizer.decode(outputs[0][context_length:], skip_special_tokens=True)
        test_dataset.loc[i,'model_reply']=reply.strip()

    #save the model reply
    test_dataset.to_json(output_path,lines=True,orient='records')
    print('save to',output_path)

    torch.cuda.empty_cache()
    return output_path

def get_model_max_length(model_path):
    if "vicuna-13b" in model_path.lower():
        return 16000-500

    elif "vicuna-7b" in model_path.lower():
        return 16000-500

    elif "llama-2-13b" in model_path.lower():
        return 4000-500

    elif "llama2-7b" in model_path.lower() or "llama-2-7b" in model_path.lower():
        return 4000-500

    elif "longchat-13b" in model_path.lower():
        return 16000-500

    elif "mpt" in model_path.lower() and "30b" in model_path.lower():
        return 8000-500

    elif "gemma" in model_path.lower() and "7b" in model_path.lower():
        return 8000-500

    elif "qwen" in model_path.lower() and "7b" in model_path.lower():
        return 20000-500

    elif "phi-3" in model_path.lower():
        return 20000-500

    elif "mistral" in model_path.lower():
        return 20000-500

    else:
        raise ValueError(f"Model {model_path} is not supported for auto max_length.")

def run_on_longbench(dataset_type,dataset_names,dataset_dir,model_path,scale_config_path,max_length=None):
    if dataset_names is not None:
        df_name_list=dataset_names
    else:
        if dataset_type is None:
            raise ValueError("dataset_type or dataset_names must be provided.")
        if dataset_type=="longbench_en":
            single_doc_datasets=["qasper","multifieldqa_en","narrativeqa"]
            multi_doc_datasets=["hotpotqa","2wikimqa","musique"]
            synthetic_datasets=["passage_retrieval_en","passage_count"]
            summary_datasets=["gov_report","multi_news",'qmsum']
            few_shot_datasets=["trec","triviaqa","samsum"]
            code_datasets=["lcc","repobench-p"]
            df_name_list=single_doc_datasets+multi_doc_datasets+synthetic_datasets+few_shot_datasets+summary_datasets

        elif dataset_type=="longbench_zh":
            multi_doc_datasets_zh=["dureader"]
            single_doc_datasets_zh=["multifieldqa_zh"]
            synthetic_datasets_zh=["passage_retrieval_zh"]
            summary_datasets_zh=["vcsum"]
            few_shot_datasets_zh=["lsht"]
            df_name_list=single_doc_datasets_zh+multi_doc_datasets_zh+synthetic_datasets_zh+summary_datasets_zh+few_shot_datasets_zh

        elif dataset_type=="longbench_e":
            multi_doc_datasets=["hotpotqa_e","2wikimqa_e"]
            single_doc_datasets=["qasper_e","multifieldqa_en_e"]
            synthetic_datasets=["passage_retrieval_en_e","passage_count_e"]
            summary_datasets=["gov_report_e","multi_news_e"]
            few_shot_datasets=["trec_e","triviaqa_e","samsum_e"]
            code_datasets=["lcc_e","repobench-p_e"]
            df_name_list=multi_doc_datasets+single_doc_datasets+synthetic_datasets+few_shot_datasets +summary_datasets+code_datasets

        else:
            raise ValueError(f"dataset_type {dataset_type} is not supported. Must be longbench_en, longbench_e or longbench_zh.")

    if max_length is None:
        max_length=get_model_max_length(model_path)

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

    # if don't apply scale dims
    if scale_config is None:
        Model = AutoModelForCausalLM
    else:
        Model = get_model_class(model_path)

    model = Model.from_pretrained(model_path,
                                  device_map="auto",
                                  trust_remote_code=True,
                                  torch_dtype="auto",
                                  attn_implementation="flash_attention_2" if not "mpt" in model_path.lower() else "eager",
                                  ).eval()

    model.config.hidden_scale_config = scale_config

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    for df_name in df_name_list:
        if not df_name.endswith(".jsonl"):
            df_name = df_name + ".jsonl"
        df_path=os.path.join(dataset_dir,df_name)
        output_path=run_on_dataset(df_path,model,tokenizer,scale_config_name,max_length)

    eval_all(os.path.dirname(output_path))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default=None)
    parser.add_argument("--dataset_names", type=str, nargs='+', default=None)

    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scale-config", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    args = parser.parse_args()
    run_on_longbench(dataset_type=args.dataset_type,
                     dataset_names=args.dataset_names,
                    dataset_dir=args.dataset_dir,
                     model_path=args.model_path,
                     scale_config_path=args.scale_config,
                     max_length=args.max_length)
