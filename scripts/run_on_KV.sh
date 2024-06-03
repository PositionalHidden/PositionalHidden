CUDA_VISIBLE_DEVICES=0 python experiments/KV/run_hidden_scale_KV.py \
--model-path "meta-llama/Llama-2-7b-chat-hf"  \
--num-kvs 140 \
--gold 0 34 69 104 139 \
--scale-config "configs/config_llama2_7b.json" \
--sample-num 100 \
--temperature 0.0
