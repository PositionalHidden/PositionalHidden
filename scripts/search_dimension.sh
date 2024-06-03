CUDA_VISIBLE_DEVICES=0 python experiments/search_dimension/search_dims.py \
--model_path "meta-llama/Llama-2-7b-chat-hf"  \
--topk_of_loss 3 \
--valid_set_path experiments/search_dimension/valid_set/KV60_valid_set.json \
--corpus_path experiments/search_dimension/corpus/random_string.txt