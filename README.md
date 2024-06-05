# Scaling Positional Hidden to Mitigates Position bias of LLMs

This repository contains the open-sourced official implementation of the paper:

[Mitigate Position Bias in Large Language Models via Scaling a Single Dimension](https://arxiv.org/abs/2406.02536) (Under Review).<br/>
_Yijiong Yu, Huiqiang Jiang, Xufang Luo, Qianhui Wu, Chin-Yew Lin, Dongsheng Li, Yuqing Yang, Yongfeng Huang and Lili Qiu_

If you find this repo helpful, please cite the following paper:

```bibtex
@article{yu2024mitigate,
  title={Mitigate Position Bias in Large Language Models via Scaling a Single Dimension},
  author={Yu, Yijiong and Jiang, Huiqiang and Luo, Xufang and Wu, Qianhui and Lin, Chin-Yew and Li, Dongsheng and Yang, Yuqing and Huang, Yongfeng and Qiu, Lili},
  journal={arXiv preprint arXiv:2406.02536},
  year={2024}
}
```

For any questions/comments, please feel free to open GitHub issues.

## 🎥 Overview

Large Language Models (LLMs) are widely used due to their generalization and generative abilities, but they exhibit position bias, especially in long-context scenarios, known as "lost in the middle." This paper explores micro-level manifestations of position bias, identifying attention weights and causal attention masks as contributing factors. We propose mitigating this bias by scaling positional hidden states. Experiments on NaturalQuestions Multi-document QA, KV retrieval, LongBench, and timeline reorder tasks, using various models (RoPE, context window-extended, and Alibi), show our method improves performance by up to 15.2% by modifying one hidden state dimension.

## 🎯 Quick Start

### Requirements

To get started with PositionalHidden, simply install it and other required dependencies using pip:

```bash
pip install -e .
pip install -r requirements.txt
```

### Step 1: Search Positional Dimensions

First, we use a prior-based Positional Hidden State Search algorithm to find the positional dimensions of the given model. The script will print the top-3 dimensions found.

```bash
bash scripts/search_dimension.sh
```

You can find the searched positional hidden states in the [`./configs`](./configs) folder.

### Step 1.5 (Optional): Visualize Positional Dimensions

To visualize the searched positional hidden states, first obtain the hidden states of the model by running the following command. The hidden states will be saved in the hidden_states directory. Then, you can visualize the positional dimensions by passing the path to the hidden states and specifying the dimension you want to visualize.

```bash
bash scripts/visualize_positional_hidden.sh
```

### Step 2: Downstream Tasks Evaluation

See [`positional_hidden/use_model.py`](positional_hidden/use_model.py) for usage, which uses llama2-7b-chat as an example. You need to manually set the dimensions to scale in the script.

#### Multi-document QA Task - NQ Dataset
Run the following command to evaluate the model on the NQ, KV, and LongBench datasets. The evaluation results will be saved in the `results` directory. To use dimension scaling, pass the `--scale-config` parameter to specify the scaling configuration file.

```bash
bash scripts/run_on_NQ.sh
```

#### Key-Value Retrieval Task
```bash
bash scripts/run_on_KV.sh
```

#### LongBench
You can download the LongBench dataset [here](https://huggingface.co/datasets/THUDM/LongBench) and place the subdatasets in the `run_on_longbench/longbench` directory. 
Then run the following command to evaluate the model on the LongBench datasets. You can choose the subdatasets by passing the `--dataset_names` or `--dataset_type` parameters, 
as shown in ```scripts/run_on_longbench.sh```:
```bash
bash scripts/run_on_longbench.sh
```
