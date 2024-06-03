import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer, AutoConfig


def get_model_class(model_path):
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


# from model_code.modeling_llama import LlamaForCausalLM
if __name__ == '__main__':
    model_path = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    AutoModelHiddenScale = get_model_class(model_path)
    model = AutoModelHiddenScale.from_pretrained(model_path,
                                                 device_map="auto",
                                                 trust_remote_code=True,
                                                 torch_dtype="auto",
                                                 attn_implementation="flash_attention_2")  # "flash_attention_2",).eval()

    # configurations of the hidden states scaling
    model.config.hidden_scale_config = {
        "dims": [2393],  # the dimensions's indices to apply the scaling
        "layers": range(10, 26),  # the layers to apply the scaling
        "factor": 0,  # the scaling factor. default is 0.
        "skip_first": 0,  # skip the first n tokens when scaling hidden states
        "last_recompute_tokens": 1,  # the number of tokens whose attention weights are recomputed
        "change_value": False,
        # whether to change the value states. If False, only the query and key states are modified
    }

    messages = [
        {"role": "user", "content": "Who are you?"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        use_cache=True,
    )
    response = outputs[0][input_ids.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))
