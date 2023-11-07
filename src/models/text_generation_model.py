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


# 読み込めるcsvファイルの大きさを増大
gb_to_bytes = 20 * 1024 * 1024 * 1024
csv.field_size_limit(gb_to_bytes)


class TextGenerationModel(object):
    def __init__(self,
                 tokenizer:str="matsuo-lab/weblab-10b",
                 model:str="matsuo-lab/weblab-10b",
                 device_type:str="cuda",
                 seed:int=19990429) -> None:
        # 出力結果保存用ディレクトリの設定
        self.date_time = get_12chars_datetime()
        self.result_dir = f"results/ja/連想語頻度表/text_generation/{self.date_time}"
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)

        # ロギングの設定
        log_path = f"{self.result_dir}/{self.date_time}.log"
        logzero.logfile(log_path)

        # シード値を固定
        logger.info(f"Seed: {seed}")
        set_seed(seed)
        random.seed(seed)

        # トークナイザーおよびモデルの読み込み
        tokenizer = model  # 怠惰
        logger.info(f"Tokenizer: {tokenizer}")
        logger.info(f"Model: {model}")

        # Please refer to `https://huggingface.co/{model}
        if model == "rinna/japanese-gpt-neox-3.6b":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
            self.model = AutoModelForCausalLM.from_pretrained(model)
        # TODO: device_map="auto" for `cyberagent/open-calm-7b` and `cyberagent/calm2-7b`
        elif model == "cyberagent/open-calm-7b" or "cyberagent/calm2-7b" or "matsuo-lab/weblab-10b":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
        elif model == "pfnet/plamo-13b":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
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
                          sample_data: List[List],
                          template: str,
                          output_path: str,
                          num_refs=3,
                          num_return_sequences=3):
        if "{references}" in template:
            logger.info(f"Number of references: {num_refs}")
        else:
            assert num_refs == 0, f"Set {num_refs} as 0"
            logger.info(f"Number of references: 0")

        # 入力用テキストの作成
        input_texts: List[str] = []
        all_references = []
        for row in sample_data:
            input_text = template
            if "{head}" in input_text:
                input_text = input_text.replace("{head}", row[0])
            if "{tail}" in input_text:
                input_text = input_text.replace("{tail}", row[1])
            if "{references}" in input_text:
                # invalid [0, 1, 2, 3] valid, v == -1: unlabled
                # references = [f"・{ref}" for i, (ref, v) in enumerate(row[-1])]
                references = [f"・{ref}" for i, (ref, v) in enumerate(row[-1]) if (int(v) == 1 or int(v) == 0)]
                random.shuffle(references)
                input_text = input_text.replace("{references}", "\n".join(references[:num_refs]))
                all_references.append(references[:num_refs])
            input_text = input_text.replace("{input_slot}", input_text)
            input_texts.append(input_text)

        # encode
        encoded_texts = self._encode_texts(input_texts)

        # テキスト生成 & 1サンプルごとにファイル出力
        logger.info(f"Number of return sequences: {num_return_sequences}")
        with open(output_path, 'w') as f:
            writer = csv.writer(f)
            with torch.no_grad():
                for i, (sample, encoded_text) in enumerate(zip(sample_data, encoded_texts)):
                    output_ids = self.model.generate(
                        encoded_text,
                        max_new_tokens=42,
                        min_new_tokens=5,
                        do_sample=True,
                        temperature=0.8,
                        pad_token_id=self.tokenizer.pad_token_id,
                        bos_token_id=self.tokenizer.bos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=num_return_sequences,
                    )
                    output_text = list(map(lambda token: self.tokenizer.decode(token, skip_special_tokens=True), output_ids))
                    if num_refs == 0:
                        writer.writerow([*sample[:-1], output_text])
                    else:
                        writer.writerow([*sample[:-1], all_references[i], output_text])
                    logger.info(f"{i+1}/{len(encoded_texts)}")

        logger.info(f"Successfully dumped {output_path} !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_type', type=str, default="cuda:0")
    parser.add_argument('--input_path', type=str, default="datasets/連想語頻度表/all/power_0.05/htrkpnsv3_30_4exp.csv.gz")
    parser.add_argument('--model', type=str, default="matsuo-lab/weblab-10b")
    parser.add_argument('--num_return_sequences', type=int, default=3)
    parser.add_argument('--num_refs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=19990429)
    parser.add_argument('--template_dir', type=str, default="datasets/連想語頻度表/templates")
    parser.add_argument('--template_name', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, default="matsuo-lab/weblab-10b")
    args = parser.parse_args()

    text_generation_model = TextGenerationModel(args.tokenizer, args.model, args.device_type, args.seed)

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
    output_path = f"{text_generation_model.result_dir}/generated_texts.csv"

    # メイン
    text_generation_model.generate_and_dump(sample_data, template, output_path, args.num_refs, args.num_return_sequences)
    logger.info("All done")

    # 生成したテキストを見やすいように整形
    result_formatter(text_generation_model.result_dir, args.num_refs, template_path, args.model)
