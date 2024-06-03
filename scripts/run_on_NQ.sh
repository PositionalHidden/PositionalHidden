## download NQ datasets
#git clone https://github.com/nelson-liu/lost-in-the-middle

CUDA_VISIBLE_DEVICES=0 python experiments/NQ/run_hidden_scale_NQ.py \
--model-path "meta-llama/Llama-2-7b-chat-hf"  \
--num-docs 20 \
--gold 0 4 9 14 19 \
--scale-config "configs/config_llama2_7b.json" \
--sample-num 500 \
--temperature 0.0