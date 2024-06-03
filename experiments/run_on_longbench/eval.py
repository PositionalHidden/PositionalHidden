import os
import json
import argparse
import numpy as np
import pandas as pd

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
    best_subspan_em,
    digit_acc
)


dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def normalize_name(name):
    if name.endswith("_e"):
        name=name[:-2]
    elif name.endswith("_e_sampled"):
        name=name[:-10]
    elif name.endswith("_e_cot"):
        name=name[:-6]

    return name

#sample the dataset to let the passage numbers distrubute evenly
def sample_passage_number(df):
    import re
    import random
    #set seed to 0
    random.seed(0)
    answers = df['answers'].tolist()
    # get passage number
    answers = [x[0] for x in answers]
    answer_nums = [int(re.findall(r"\d+", i)[0]) for i in answers]
    df["answer_digit"] = answer_nums
    answer_nums_0_10 = [i for i in answer_nums if 1 <= i <= 10]
    answer_nums_10_20 = [i for i in answer_nums if 10 < i <= 20]
    answer_nums_20_30 = [i for i in answer_nums if 20 < i <= 30]
    print("0-5:", len(answer_nums_0_10) / len(answer_nums))
    print("10-20:", len(answer_nums_10_20) / len(answer_nums))
    print("20-30:", len(answer_nums_20_30) / len(answer_nums))

    df["answer_digit_group"] = pd.cut(df["answer_digit"], bins=[0, 6, 12, 18, 24, 30], right=False)
    df_grouped = df.groupby("answer_digit_group")
    for name, group in df_grouped:
        print(name, len(group))

    # let the number of each group be the same
    min_group_num = df_grouped.size().min()
    df_sampled = pd.concat([group.sample(min_group_num,random_state=0) for name, group in df_grouped])

    return df_sampled

def get_file_paths(dir,suffix,subfolder=True,exclude_suffix=None):
    file_path_list=[]
    if subfolder==False:
        for file in os.listdir(dir):
            if file.endswith(suffix):
                file_path_list.append(os.path.join(dir, file))
    else:
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(suffix):
                    file_path_list.append(os.path.join(root, file))
    if exclude_suffix!=None:
        file_path_list=[file_path for file_path in file_path_list if not file_path.endswith(exclude_suffix)]

    return file_path_list


def scorer_e(dataset, predictions, answers, lengths, all_classes)->dict:
    dataset= normalize_name(dataset)

    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes)->float:
    dataset= normalize_name(dataset)

    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

def eval_all(df_dir):
    import pandas as pd
    df_path_list=get_file_paths(df_dir,suffix=".jsonl",subfolder=False)
    scores={}
    for df_path in df_path_list:
        dataset_name=os.path.basename(df_path).replace(".jsonl","")
        df=pd.read_json(df_path,lines=True)

        #passage_retrieval_en_e is imbalanced, so we sample it
        if dataset_name=="passage_retrieval_en_e":
            df=sample_passage_number(df)
            dataset_name="passage_retrieval_en_e_sampled"

        predictions=df['model_reply'].tolist()
        answers=df['answers'].tolist()
        lengths=df['length'].tolist()
        all_classes=df['all_classes'].tolist()
        score = scorer(dataset_name, predictions, answers, all_classes[0])
        scores[dataset_name] = score

    print(df_dir)
    import pprint
    pprint.pprint(scores)

    #print the average score
    print("Avg: ",np.mean(list(scores.values())))
    scores_of_tasks=split_by_type(scores)

    #save
    with open(df_dir+"/scores_of_datasets.json", "w") as f:
        json.dump(scores, f)
    with open(df_dir+"/scores_of_tasks.json", "w") as f:
        json.dump(scores_of_tasks, f)

    return scores

#calculate the average score of each task type
def split_by_type(scores:dict):
    from pprint import pprint
    single_doc_datasets = ["qasper", "multifieldqa_en", "narrativeqa"]
    multi_doc_datasets = ["hotpotqa", "2wikimqa", "musique"]
    synthetic_datasets = ["passage_retrieval_en", "passage_count"]
    summary_datasets = ["gov_report", "multi_news", 'qmsum']
    few_shot_datasets = ["trec", "triviaqa", "samsum"]
    code_datasets = ["lcc", "repobench-p"]
    single_doc_scores = {}
    multi_doc_scores = {}
    synthetic_scores = {}
    summary_scores = {}
    few_shot_scores = {}
    code_scores = {}
    for key in scores.keys():
        if "qasper" in key or "multifieldqa" in key or "narrativeqa" in key:
            single_doc_scores[key] = scores[key]
        elif "hotpotqa" in key or "2wikimqa" in key or "musique" in key or 'dureader' in key:
            multi_doc_scores[key] = scores[key]
        elif "passage_retrieval" in key or "passage_count" in key:
            synthetic_scores[key] = scores[key]
        elif "gov_report" in key or "multi_news" in key or 'qmsum' in key or "vcsum" in key:
            summary_scores[key] = scores[key]
        elif "trec" in key or "triviaqa" in key or "samsum" in key or "lsht" in key:
            few_shot_scores[key] = scores[key]
        elif "lcc" in key or "repobench-p" in key:
            code_scores[key] = scores[key]

    #calculate the average score of each task type
    single_doc_scores_mean=np.round(np.mean(list(single_doc_scores.values())),2) if len(single_doc_scores)>0 else 0
    multi_doc_scores_mean=np.round(np.mean(list(multi_doc_scores.values())),2) if len(multi_doc_scores)>0 else 0
    synthetic_scores_mean=np.round(np.mean(list(synthetic_scores.values())),2) if len(synthetic_scores)>0 else 0
    summary_scores_mean=np.round(np.mean(list(summary_scores.values())),2) if len(summary_scores)>0 else 0
    few_shot_scores_mean=np.round(np.mean(list(few_shot_scores.values())),2) if len(few_shot_scores)>0 else 0
    code_scores_mean=np.round(np.mean(list(code_scores.values())),2) if len(code_scores)>0 else 0

    print("single_doc_scores:")
    pprint(single_doc_scores)
    print("multi_doc_scores:")
    pprint(multi_doc_scores)
    print("few_shot_scores:")
    pprint(few_shot_scores)
    print("synthetic_scores:")
    pprint(synthetic_scores)
    print("summary_scores:")
    pprint(summary_scores)
    print("code_scores:")
    pprint(code_scores)

    print("single_doc_scores_mean:",single_doc_scores_mean)

    print("multi_doc_scores_mean:",multi_doc_scores_mean)

    print("synthetic_scores_mean:",synthetic_scores_mean)

    print("summary_scores_mean:",summary_scores_mean)

    print("few_shot_scores_mean:",few_shot_scores_mean)

    print("code_scores_mean:",code_scores_mean)

    mean_scores=[single_doc_scores_mean,multi_doc_scores_mean,synthetic_scores_mean,summary_scores_mean,few_shot_scores_mean,code_scores_mean]
    all_scores_mean=np.round(np.mean([score for score in mean_scores if score>0]),2)
    print("Avg Score:",all_scores_mean)

    scores_of_tasks={
                      "single_doc_scores_mean":single_doc_scores_mean,
                      "multi_doc_scores_mean":multi_doc_scores_mean,
                      "few_shot_scores_mean":few_shot_scores_mean,
                      "synthetic_scores_mean":synthetic_scores_mean,
                      "summary_scores_mean":summary_scores_mean,
                      "code_scores_mean":code_scores_mean,
                      "all_scores_mean":all_scores_mean
                      }

    return scores_of_tasks


if __name__ == '__main__':


    eval_all("./model_responses/llama2-7b-chat/scale_no_scale")








