import argparse
import os

from get_avg_hidden_state import get_hidden_states
from seach_dim_by_statistic import search_by_statistic
from eval_on_valid_set import eval_dims_main


def main(model_path, corpus_path, sample_num, max_length, add_bos_token, topk_of_smooth, topk_of_loss, valid_set_path):
    # get hidden states
    hidden_states, hidden_states_save_path = get_hidden_states(model_path, corpus_path, sample_num, max_length,
                                                               add_bos_token, save_dir="./hidden_states")
    # search by statistic
    dim_cand = search_by_statistic(hidden_states, topk_of_smooth)
    # eval on valid set
    dim_cand2 = eval_dims_main(model_path, dataset_path=valid_set_path, dim_cand=dim_cand, topk=topk_of_loss)

    print("the positional dimensions of {} are {}".format(model_path, str(dim_cand2)))

    return dim_cand2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, default="./corpus/random_string.txt")
    parser.add_argument("--sample_num", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=1000)
    parser.add_argument("--add_bos_token", type=bool, default=False)
    parser.add_argument("--topk_of_smooth", type=int, default=10)
    parser.add_argument("--topk_of_loss", type=int, default=3)
    parser.add_argument("--valid_set_path", type=str, default="./valid_set/KV60_valid_set.json")
    args = parser.parse_args()
    result=main(args.model_path,
         args.corpus_path,
         args.sample_num,
         args.max_length,
         args.add_bos_token,
         args.topk_of_smooth,
         args.topk_of_loss,
         args.valid_set_path)

