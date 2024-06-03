cd experiments/seach_dimension
CUDA_VISIBLE_DEVICES=0 python search_dims.py \
--model_path "meta-llama/Llama-2-7b-chat-hf" \
--topk_of_loss 3