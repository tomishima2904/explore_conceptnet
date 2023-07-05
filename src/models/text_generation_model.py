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

from src.utils.file_handlers import get_12chars_datetime


# 読み込めるcsvファイルの大きさを増大
gb_to_bytes = 20 * 1024 * 1024 * 1024
csv.field_size_limit(gb_to_bytes)


class TextGenerationModel(object):
    def __init__(self,
                 tokenizer:str="rinna/japanese-gpt-neox-3.6b",
                 model:str="rinna/japanese-gpt-neox-3.6b",
                 device_type:str="cuda") -> None:
        # 出力結果保存用ディレクトリの設定
        self.date_time = get_12chars_datetime()
        self.result_dir = f"results/ja/連想語頻度表/text_generation/{self.date_time}"
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)

        # ロギングの設定
        log_path = f"{self.result_dir}/{self.date_time}.log"
        logzero.logfile(log_path)

        # シード値を固定
        set_seed(19990429)

        # トークナイザーおよびモデルの読み込み
        logger.info(f"Tokenizer: {tokenizer}")
        logger.info(f"Model: {model}")
        if model == "rinna/japanese-gpt-neox-3.6b":  # https://huggingface.co/rinna/japanese-gpt-neox-3.6b
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
            self.model = AutoModelForCausalLM.from_pretrained(model)
        # TODO: device_map="auto"
        elif model == "cyberagent/open-calm-7b":  # https://huggingface.co/cyberagent/open-calm-7b
            self.tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-7b")
            self.model = AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-7b", torch_dtype=torch.float16)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.model = AutoModelForCausalLM.from_pretrained(model)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.to(device_type)
        logger.info(f"Device: {self.model.device}")


    def _encode_texts(self, texts: List[str]):
        return [self.tokenizer.encode(text,
                                      add_special_tokens=False,
                                      return_tensors="pt").to(self.model.device)
                for text in texts]


    def generate_and_dump(self,
                          sample_data: List[str],
                          template: str,
                          output_path: str,
                          num_refs=3,
                          num_return_sequences=3):

        # 入力用テキストの作成
        input_texts: List[str] = []
        for row in sample_data:
            input_text = template
            if "{words_set}" in input_text:
                words_set = f"{row[0]}, {row[1]}"
                input_text = input_text.replace("{words_set}", words_set)
            if "{references}" in input_text:
                references = [f"- {ref}" for i, ref in enumerate(row[-1])]
                input_text = input_text.replace("{references}", "\n".join(references[:num_refs]))
            input_text = input_text.replace("{input_slot}", input_text)
            input_texts.append(input_text)

        # encode
        encoded_texts = text_generation_model._encode_texts(input_texts)

        # テキスト生成 & 1サンプルごとにファイル出力
        logger.info(f"Number of return sequences: {num_return_sequences}")
        with open(output_path, 'w') as f:
            writer = csv.writer(f)
            with torch.no_grad():
                for i, (sample, encoded_text) in enumerate(zip(sample_data, encoded_texts)):
                    output_ids = self.model.generate(
                        encoded_text,
                        max_new_tokens=100,
                        min_new_tokens=5,
                        do_sample=True,
                        temperature=0.8,
                        pad_token_id=self.tokenizer.pad_token_id,
                        bos_token_id=self.tokenizer.bos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=num_return_sequences,
                    )
                    output_text = list(map(lambda token: self.tokenizer.decode(token, skip_special_tokens=True), output_ids))
                    writer.writerow([*sample[:2], output_text])
                    logger.info(f"{i+1}/{len(encoded_texts)}")

        logger.info(f"Successfully dumped {output_path} !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_type', type=str, default="cuda:0")
    parser.add_argument('--input_path', type=str, default="datasets/連想語頻度表/pairs/htns_200_best10_pairs.csv.gz")
    parser.add_argument('--model', type=str, default="rinna/japanese-gpt-neox-3.6b")
    parser.add_argument('--num_refs', type=int, default=3)
    parser.add_argument('--template_dir', type=str, default="datasets/連想語頻度表/templates")
    parser.add_argument('template_name', type=str)
    parser.add_argument('--tokenizer', type=str, default="rinna/japanese-gpt-neox-3.6b")
    args = parser.parse_args()

    text_generation_model = TextGenerationModel(args.tokenizer, args.model, args.device_type)

    # 刺激語と連想語と抽出文数と抽出文からなるデータを読み込む
    logger.info(f"Dataset: {args.input_path}")
    with gzip.open(args.input_path, 'rt') as f:
        reader = csv.reader(f)
        all_data = [[*row[:2], int(row[2]), eval(row[-1])] for row in reader]

    # 使用するデータをサンプリング
    sample_data = [row for row in all_data if row[2]>2]
    """sample_data
    head: str
    tail: str
    sentences: List[str]
    """

    # プロンプト入力用のテンプレートを読み込む
    template_path = f"{args.template_dir}/{args.template_name}.json"
    logger.info(f"Template: {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
            template = json.load(f)["prompt_input"]

    # 出力ファイル名を命名
    if args.model == "rinna/japanese-gpt-neox-3.6b":
        model_type = "rinna3.6b"
    elif args.model == "cyberagent/open-calm-7b":
        model_type = "calm7b"
    else:
        model_type = "else"
    output_path = f"{text_generation_model.result_dir}/{model_type}_{args.template_name}.csv"

    logger.info(f"Number of references: {args.num_refs}")
    text_generation_model.generate_and_dump(sample_data, template, output_path, args.num_refs)

    logger.info("All done")
