CUDA_VISIBLE_DEVICES=0 python experiments/run_on_longbench/run_on_longbench.py \
--dataset_names "trec_e" "lcc_e" \
--dataset_dir "./run_on_longbench/longbench" \
--model_path "meta-llama/Llama-2-7b-chat-hf" \
--scale-config "./configs/config_llama2_7b.json" 

CUDA_VISIBLE_DEVICES=0 python experiments/run_on_longbench/run_on_longbench.py \
--dataset_type "longbench_en" \
--dataset_dir "./run_on_longbench/longbench" \
--model_path "meta-llama/Llama-2-7b-chat-hf" \
--scale-config "./configs/config_llama2_7b.json" 