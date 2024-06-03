import torch
from tqdm import tqdm
import numpy as np
import pathlib

import os
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_hidden_states(model_path, corpus_path, sample_num, max_length, add_bos_token=False, save_dir=None, new=False):
    model_name = os.path.basename(model_path)
    corpus_name = os.path.basename(corpus_path).replace(".txt", "")

    if save_dir is not None:
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = save_dir + "/hidden_states_{}_on_{}.npy".format(model_name, corpus_name)
    else:
        save_path = None

    if not new and save_path is not None and os.path.exists(save_path):
        hidden_states_mean = np.load(save_path).astype(np.float32)
        print("load hidden_states from {}".format(save_path))
        return hidden_states_mean, save_path

    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True,
                                              add_bos_token=add_bos_token,
                                              add_eos_token=False)

    if "nope" in model_path.lower():
        from transformers.models.llama import modeling_llama
        def nope_monkey_patch(q, k, *args, **kwargs):
            return q, k

        modeling_llama.apply_rotary_pos_emb = nope_monkey_patch

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map="auto",
                                                 trust_remote_code=True,
                                                 attn_implementation="flash_attention_2" if "mpt" not in model_path.lower() else "eager",
                                                 torch_dtype="auto",
                                                 ).eval()
    config = model.config

    # get the corpus
    with open(corpus_path, "r") as f:
        corpus = f.read()
    corpus_sharded = [corpus[i:i + max_length * 10] for i in range(0, len(corpus), max_length * 10)]

    if "mpt" in model_path.lower():
        config.num_hidden_layers = config.n_layers
        config.hidden_size = config.d_model

    # initialize the hidden states
    hidden_states_sum = torch.zeros([config.num_hidden_layers + 1, max_length, config.hidden_size],
                                    dtype=torch.float16).cuda()
    model.requires_grad_(False)
    model.config.use_cache = False

    for i in tqdm(range(sample_num), desc="use corpus to get hidden states"):
        text = corpus_sharded[i]

        input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors=None)

        # input_ids are truncated to max_length
        input_ids = input_ids[:max_length]
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)

        with torch.no_grad():
            with torch.autocast("cuda"):
                output = model(input_ids=input_ids, output_hidden_states=True, output_attentions=False)

        # record the hidden states
        hidden_states = torch.stack(output.hidden_states, dim=0)
        hidden_states = torch.mean(hidden_states, dim=1)  # average over the batch
        hidden_states_sum = hidden_states_sum + hidden_states

    del model, output, hidden_states

    # average the hidden states
    hidden_states_mean = hidden_states_sum.cpu().float().numpy() / sample_num

    np.save(save_path, hidden_states_mean) if save_path is not None else None
    print("save hidden_states to {}".format(save_path))
    return hidden_states_mean, save_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="model path")
    parser.add_argument("--corpus_path", type=str, required=True, help="corpus path")
    parser.add_argument("--sample_num", type=int, default=250, help="sample number")
    parser.add_argument("--max_length", type=int, default=1000, help="max length")

    args = parser.parse_args()
    # model_path = "meta-llama/Llama-2-7b-chat-hf"
    # corpus_path = "../random_string.txt"

    hidden_states = get_hidden_states(
        model_path=args.model_path,
        corpus_path=args.corpus_path,
        sample_num=args.sample_num,
        max_length=args.max_length,
        add_bos_token=False,
        save_dir="./hidden_states")
