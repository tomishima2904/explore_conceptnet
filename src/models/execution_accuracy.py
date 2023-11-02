import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

import re
import csv
import gzip
import json
import logzero
from logzero import logger
from typing import Any, Dict, Optional, List, Union
import os
import argparse
import random
import numpy as np

from src.utils.result_formatter import completion_formatter


class CompletionsRater(object):
    def __init__(self,
                 tokenizer:str="rinna/japanese-roberta-base",
                 model:str="rinna/japanese-roberta-base",
                 seed:int=19990429) -> None:
        # シード値を固定
        logger.info(f"Seed: {seed}")
        set_seed(seed)
        random.seed(seed)

        # トークナイザーおよびモデルの読み込み
        tokenizer = model  # 怠惰
        logger.info(f"Tokenizer: {tokenizer}")
        logger.info(f"Model: {model}")
        if model == "rinna/japanese-roberta-base":  # https://huggingface.co/rinna/japanese-roberta-base
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
            self.tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
            self.model = AutoModelForCausalLM.from_pretrained(model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.model = AutoModelForCausalLM.from_pretrained(model)
        self.model.eval()


    # Excution Accuracy の結果と その逆順位を返す
    def execution_accuracy(self,
                           text: str,
                           head: str,
                           tail: str,
                           topk=100,
                           out_of_rank:int=None,
                           out_of_sentence:int=None):
        # 各種言語モデルに合わせてテキストに特殊トークンを付与
        start_of_text_token = "[CLS]"
        text = start_of_text_token + text

        # Tokenize and Convert tokens to IDs
        tokens = self.tokenizer.tokenize(text)
        token_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(tokens))

        # head と tail をそれぞれ [MASK] にする
        head_indices = [i for i, t in enumerate(tokens) if t == head]
        tail_indices = [i for i, t in enumerate(tokens) if t == tail]
        head_masked_ids = token_ids.detach().clone()
        tail_masked_ids = token_ids.detach().clone()
        head_masked_ids[head_indices] = self.tokenizer.mask_token_id
        tail_masked_ids[tail_indices] = self.tokenizer.mask_token_id
        position_ids = torch.LongTensor(list(range(0, head_masked_ids.size(-1))))

        # MRRを計算しやすくするようにするための工夫
        # topk以内に正解語が存在しない場合，順位としてtopkを割り当てておく
        if not out_of_rank:
            out_of_rank = topk
        # 文中に特定の語が存在しない場合，順位としてtopk*10を割り当てておく
        if not out_of_sentence:
            out_of_sentence = topk * 10

        # Execution Accuracy を計算
        with torch.no_grad():
            head_masked_outputs = self.model(input_ids=head_masked_ids.unsqueeze(0), position_ids=position_ids)
            tail_masked_outputs = self.model(input_ids=tail_masked_ids.unsqueeze(0), position_ids=position_ids)

            head_rank = out_of_rank  # topk内に正解語がない場合, その順位はtopkとなる.
            head_rr = 0  # rr = reciprocal rank (逆順位: 順位の逆数)

            # headのトークンがcompletion内に存在しない場合，初期順位にはout_of_sentenceを与える
            if len(head_indices) == 0:
                head_rank = out_of_sentence
            # headのトークンがcompletion内に存在する場合，初期順位にはout_of_rankを与える
            else:
                head_rank = out_of_rank
            for head_index in head_indices:
                predictions = head_masked_outputs[0][0, head_index].topk(topk)
                for i, index_t in enumerate(predictions.indices):
                    index = index_t.item()
                    token = self.tokenizer.convert_ids_to_tokens([index])[0]
                    if token == head or head_rank == i:
                        head_rank = i
                        head_rr = 1/(head_rank+1)
                        break
            if head_rank >= out_of_sentence:
                head_rr = -1

            tail_rank = out_of_rank
            tail_rr = 0
            if len(tail_indices) == 0:
                tail_rank = out_of_sentence
            else:
                tail_rank = out_of_rank
            for tail_index in tail_indices:
                predictions = tail_masked_outputs[0][0, tail_index].topk(topk)
                for i, index_t in enumerate(predictions.indices):
                    index = index_t.item()
                    token = self.tokenizer.convert_ids_to_tokens([index])[0]
                    if token == tail or tail_rank == i:
                        tail_rank = i
                        tail_rr = 1/(tail_rank+1)
                        break
            if tail_rank >= out_of_sentence:
                tail_rr = -1

        return head_rank, tail_rank, head_rr, tail_rr


    def rate_completions(self, input_data: List):
        """input_data
        0) relation: str (Not used)
        1) head: str
        2) tail: str
        3) completions: List[str]
        """
        rated_result = []
        for row in input_data:
            trimmed_completions = []
            ranks = []
            rrs = []
            for completion in row[-1]:
                trimmed_completion = completion_formatter(completion)
                trimmed_completions.append(trimmed_completion)
                head_rank, tail_rank, head_rr, tail_rr = \
                    completion_rater.execution_accuracy(trimmed_completion, row[1], row[2], args.topk)
                ranks.append((head_rank, tail_rank))
                rrs.append((head_rr, tail_rr))

                # ランクの昇順に出力をソートする場合 (しないほうがいいかも)
                # sorted_ranks_indices = sorted(range(len(ranks)), key=lambda i: ranks[i][0]+ranks[i][1])
                # ranks = [ranks[i] for i in sorted_ranks_indices]
                # trimmed_completions = [trimmed_completions[i] for i in sorted_ranks_indices]

            rated_result.append([*row[:-1], trimmed_completions ,ranks, rrs])
            
        return rated_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default="datasets/連想語頻度表/all/power_0.05/htrkpnsv3_30_4exp.csv.gz")
    parser.add_argument('--model', type=str, default="rinna/japanese-roberta-base")
    parser.add_argument('--seed', type=int, default=19990429)
    parser.add_argument('--tokenizer', type=str, default="rinna/japanese-roberta-base")
    parser.add_argument('--topk', type=int, default=100)
    args = parser.parse_args()

    # GPTからの出力結果 (head, tail, completions) を読み込む
    logger.info(f"Result dir: {args.result_dir}")
    input_path = f"{args.result_dir}/formatted_results.csv"
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        input_data = [[row[1], row[2], row[3], eval(row[4])] for row in reader]  # (head, tail, completions)

    completion_rater = CompletionsRater(args.tokenizer, args.model, args.seed)

    # Masked Language Model でtextの評価を行い，その結果をcsv出力
    rated_result = completion_rater.rate_completions(input_data)
    output_path = f"{args.result_dir}/rated_results.csv"
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rated_result)
    logger.info(f"Successfully dumped {output_path}!")

