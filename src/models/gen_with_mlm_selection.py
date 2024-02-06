import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

import csv
import gzip
import json
import logzero
from logzero import logger
from typing import Any, Dict, Optional, List, Union
import os
import argparse
import random

from src.utils.file_handlers import get_12chars_datetime
from src.utils.result_formatter import result_formatter
from src.models.text_generation_model import TextGenerationModel
from src.models.execution_accuracy import CompletionsRater


# 読み込めるcsvファイルの大きさを増大
gb_to_bytes = 20 * 1024 * 1024 * 1024
csv.field_size_limit(gb_to_bytes)


# formatted_results.csvと同じ形式のデータを返す
# m_and_a_comparator.merge_m_and_a_resultsを参考
def sort_by_mlm_score(input_data: List,
                      output_path: str,  # formatted_results.csv
                      output_path_all: str,  # formmatted_results_detailed.csv
                      intra_selection_option = "softmax",
                      num_all_sequences=30,
                      num_return_sequences=3,
                      num_pairs=50):
    """input_data
    0) rel: str
    1) head: str
    2) tail: str
    3) completions: List[str]
    4) ranks: List[Tuple(int, int)]
    5) rrs: List[Tuple(float, float)]
    6) scores: List[Tuple(float, float)]
    """

    """output_results (row in formatted_results.csv)
    0) id: int
    1) rel: str
    2) head: str
    3) tail: str
    4) completions: List[str], len(completions) == num_return_sequences
    """

    """all_output_results (row in formmatted_results_detailed.csv)
    0) rel: str
    1) head: str
    2) tail: str
    3) completions: List[str], len(completions) == num_all_sequences
    4) ranks: List[Tuple(int, int)]
    5) rrs: List[Tuple(float, float)]
    6) scores: List[Tuple(float, float)]
    """

    with open(output_path, 'w') as wf, open(output_path_all, 'w') as wf_all:
        writer = csv.writer(wf)
        writer_all = csv.writer(wf_all)
        for i in range(num_pairs):
            rel, head, tail = input_data[i][:3]
            completions = input_data[i][3]
            completions = [completion for completion in eval(input_data[i][3])]
            ranks = [rank for rank in eval(input_data[i][4])]
            rrs = [rr for rr in eval(input_data[i][5])]
            scores = [score for score in eval(input_data[i][6])]
            
            # MRRの和を第1キーに，softmax_scoreの和を第2キーにして，全て降順にソートする
            if intra_selection_option == "softmax":
                sorted_rrs_indices = sorted(range(num_all_sequences),
                                            key=lambda j: (rrs[j][0]+rrs[j][1],
                                                           scores[j][0]+scores[j][1]),
                                            reverse=True)
            else:
                sorted_rrs_indices = sorted(range(num_all_sequences),
                                            key=lambda j: (rrs[j][0]+rrs[j][1],
                                                           1/len(completions[j]),
                                                           scores[j][0]+scores[j][1]),
                                            reverse=True)
            sorted_completions = [completions[j] for j in sorted_rrs_indices]
            sorted_ranks = [ranks[j] for j in sorted_rrs_indices]
            sorted_rrs = [rrs[j] for j in sorted_rrs_indices]
            sorted_scores = [scores[j] for j in sorted_rrs_indices]

            # 結果を出力
            output_row = [rel, head, tail, sorted_completions, sorted_ranks, sorted_rrs, sorted_scores]
            writer_all.writerow(output_row)
            writer.writerow([i, rel, head, tail, sorted_completions[:num_return_sequences]])
                
    print(f"Successfully dumped {output_path}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_type', type=str, default="cuda:0")
    parser.add_argument('--dir_name', type=str, default="text_generation")
    parser.add_argument('--input_path', type=str, default="datasets/連想語頻度表/all/power_0.05/htrkpnsv3_30_4exp.csv.gz")
    parser.add_argument('--model', type=str, default="matsuo-lab/weblab-10b")
    parser.add_argument('--num_pre_return_sequences', type=int, default=30)
    parser.add_argument('--num_return_sequences', type=int, default=3)
    parser.add_argument('--num_refs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=19990429)
    parser.add_argument('--template_dir', type=str, default="datasets/連想語頻度表/templates")
    parser.add_argument('--template_name', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, default="matsuo-lab/weblab-10b")
    args = parser.parse_args()

    text_generation_model = TextGenerationModel(args.tokenizer, args.model, args.device_type, args.seed, args.dir_name)

    # 刺激語と連想語と抽出文数と抽出文からなるデータを読み込む
    logger.info(f"Dataset: {args.input_path}")
    with gzip.open(args.input_path, 'rt') as f:
        reader = csv.reader(f)
        all_data = [[*row[:-2], int(row[-2]), eval(row[-1])] for row in reader]

    # 使用するデータをサンプリング
    sample_data = [row for row in all_data]
    """sample_data
    head: str
    tail: str
    relations: List[str]
    k (rank): int
    power: float
    num_sentences: int
    sentences: List[str]
    """

    # プロンプト入力用のテンプレートを読み込む
    template_path = f"{args.template_dir}/{args.template_name}.json"
    logger.info(f"Template: {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
            template = json.load(f)["prompt_input"]

    # 出力ファイル名を命名
    output_path = f"{text_generation_model.result_dir}/tmp_generated_texts.csv"

    # 連想理由を生成し，出力
    text_generation_model.generate_and_dump(sample_data, template, output_path, args.num_refs, args.num_pre_return_sequences)
    logger.info("All done")

    # 生成したテキストを見やすいように整形
    input_path = f"{text_generation_model.result_dir}/tmp_generated_texts.csv"
    output_path_csv = f"{text_generation_model.result_dir}/tmp_formatted_results.csv"
    output_path_txt = f"{text_generation_model.result_dir}/tmp_formatted_results.txt"
    result_formatter(input_path, output_path_txt, output_path_csv, args.num_refs, template_path, args.model)

    # 生成した連想理由に対しMLMでスコアリングするためのデータを再ロード
    input_path = f"{text_generation_model.result_dir}/tmp_formatted_results.csv"
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        input_data = [[row[1], row[2], row[3], eval(row[4])] for row in reader]  # (head, tail, completions)

    completion_rater = CompletionsRater(args.tokenizer, args.model, args.seed)

    # Masked Language Model でtextの評価を行い，その結果をcsv出力
    rated_result = completion_rater.rate_completions(input_data)
    output_path = f"{text_generation_model.result_dir}/rated_results.csv"
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rated_result)
    logger.info(f"Successfully dumped {output_path}!")

    # MLMで評価したデータを読み込む
    input_path = f"{text_generation_model.result_dir}/rated_results.csv"
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        input_data = [line for line in reader]
    output_path = f"{text_generation_model.result_dir}/formatted_results.csv"
    output_path_all = f"{text_generation_model.result_dir}/formatted_results_detailed.csv"

    # 生成した連想理由に対し，提案手法でスコアリングをして，出力する連想理由を精選
    sort_by_mlm_score(input_data,
                      output_path,
                      output_path_all,
                      args.intra_selection_option,
                      args.num_pre_return_sequences,
                      args.num_return_sequences)
