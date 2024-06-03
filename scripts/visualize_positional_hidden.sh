cd experiments/seach_dimension

CUDA_VISIBLE_DEVICES=0 python get_avg_hidden_state.py \
--model_path "meta-llama/Llama-2-7b-chat-hf" \
--corpus_path "./corpus/random_string.txt" \
--sample_num 200 \
--max_length 1000 

python visualize_positional_hidden.py \
--hidden_states_path "./hidden_states/hidden_states_llama2-7b-chat_on_random_string.npy" \
--dim 2393 \
--output_dir "./visual_llama2-7b" \
--skip_first 30
