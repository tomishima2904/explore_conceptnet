import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

import csv
import gzip
import json
import logzero
from logzero import logger
from typing import Any, Dict, Optional, List, Union
import os

from src.utils.file_handlers import get_12chars_datetime


# 読み込めるcsvファイルの大きさを増大
gb_to_bytes = 20 * 1024 * 1024 * 1024
csv.field_size_limit(gb_to_bytes)


class TextGenerationModel(object):
    def __init__(self,
                 tokenizer:str="rinna/japanese-gpt-neox-3.6b",
                 model:str="rinna/japanese-gpt-neox-3.6b") -> None:
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
            self.model = self.model.to("cuda:0")
        logger.info(f"Device: {self.model.device}")


    def encode_texts(self, texts: List[str]):
        return [self.tokenizer.encode(text,
                                      add_special_tokens=False,
                                      return_tensors="pt").to(self.model.device)
                for text in texts]


    def generate_texts(self, encoded_texts: List[torch.Tensor], num_return_sequences=3) -> list:
        """ GPTでテキスト生成
        encoded_texts: [torch.Tensor] -> [str]
        """
        logger.info(f"Number of return sequences: {num_return_sequences}")
        with torch.no_grad():
            output_ids_list = []
            for i, encoded_text in enumerate(encoded_texts):
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
                output_ids_list.append(output_ids)
                logger.info(f"{i+1}/{len(encoded_texts)}")

        # モデルからの出力をデコード
        output_texts = []
        for output_ids in output_ids_list:
            output_text = list(map(lambda token: self.tokenizer.decode(token, skip_special_tokens=True), output_ids))
            output_texts.append(output_text)
        return output_texts


    def dump_result(self, output_file:str,
                    output_texts:List[str],
                    sample_words:List[List[str]]):
        output_path = f"{self.result_dir}/{output_file}"
        with open(output_path, 'w') as f:
            writer = csv.writer(f)
            for pair, text in zip(sample_words, output_texts):
                writer.writerow([pair[0], pair[1], text])
        logger.info(f"Successfully dumped {output_path} !")


if __name__ == "__main__":

    tokenizer = "rinna/japanese-gpt-neox-3.6b"
    model = "rinna/japanese-gpt-neox-3.6b"
    text_generation_model = TextGenerationModel(tokenizer, model)

    # 刺激語と連想語と抽出文数と抽出文からなるデータを読み込む
    input_dir = "datasets/連想語頻度表/pairs"
    input_path = f"{input_dir}/htns_200_best10_pairs.csv.gz"
    logger.info(f"Dataset: {input_path}")
    with gzip.open(input_path, 'rt') as f:
        reader = csv.reader(f)
        all_data = [[*row[:2], int(row[2]), eval(row[-1])] for row in reader]

    # 使用するデータをサンプリング
    sample_data = [row for row in all_data if row[2]>2]
    sample_pairs = [[*row[:2]] for row in sample_data]

    # プロンプト入力用のテンプレートを読み込む
    template_dir = "datasets/連想語頻度表/templates"
    template_name = "one-shot_no_refs"
    template_path = f"{template_dir}/{template_name}.json"
    logger.info(f"Template: {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
        template = json.load(f)["prompt_input"]

    # 入力用テキストの作成
    input_texts = []
    for row in sample_data:
        input_text = template
        if "{words_set}" in input_text:
            words_set = f"{row[0]}, {row[1]}"
            input_text = input_text.replace("{words_set}", words_set)
        if "{references}" in input_text:
            references = [f"- {ref}" for i, ref in enumerate(row[-1])]
            input_text = input_text.replace("{references}", "\n".join(references))
        input_text = input_text.replace("{input_slot}", input_text)
        input_texts.append(input_text)

    encoded_texts = text_generation_model.encode_texts(input_texts)
    output_texts = text_generation_model.generate_texts(encoded_texts)

    if model == "rinna/japanese-gpt-neox-3.6b":
        model_type = "rinna3.6b"
    elif model == "cyberagent/open-calm-7b":
        model_type = "calm7b"
    else:
        model_type = "else"
    output_file = f"{model_type}_{template_name}.csv"
    text_generation_model.dump_result(output_file, output_texts, sample_pairs)

    logger.info("All done")
