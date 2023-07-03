import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

import csv
import gzip
import json
import logzero
from logzero import logger

from src.utils.file_handlers import get_12chars_datetime


# 読み込めるcsvファイルの大きさを増大
gb_to_bytes = 20 * 1024 * 1024 * 1024
csv.field_size_limit(gb_to_bytes)


class RinnaJapaneseGpt(object):
    def __init__(self) -> None:
        # ロギングの設定
        self.date_time = get_12chars_datetime()
        log_path = f"logs/{self.date_time}.log"
        logzero.logfile(log_path)

        # シード値を固定
        set_seed(19990429)

        # トークナイザーおよびモデルの読み込み
        logger.info("Loading model ...")
        self.tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b", use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b")
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")


    def encode_texts(self, texts: list):
        return [self.tokenizer.encode(text,
                                      add_special_tokens=False,
                                      return_tensors="pt")
                for text in texts]


    def generate_texts(self, encoded_texts: list, num_return_sequences=3) -> list:
        """ GPTでテキスト生成
        encoded_texts: [torch.Tensor] -> [str]
        """
        with torch.no_grad():
            output_ids_list = []
            for i, encoded_text in enumerate(encoded_texts):
                output_ids = self.model.generate(
                    encoded_text.to(self.model.device),
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
        for output_sentences in output_ids_list:
            output_texts_per_sequence = []
            for output_sentence in output_sentences:
                # [PAD]トークンを除外
                filtered_output = [token for token in output_sentence
                                   if token != self.tokenizer.pad_token_id]
                decoded_output = self.tokenizer.decode(filtered_output)
                output_texts_per_sequence.append(decoded_output)
            output_texts.append(output_texts_per_sequence)
        return output_texts


    def dump_result(self, output_dir:str, output_texts:list, sample_words:list):
        output_path = f"{output_dir}/{self.date_time}.csv"
        with open(output_path, 'w') as f:
            writer = csv.writer(f)
            for pair, text in zip(sample_words, output_texts):
                writer.writerow([pair[0], pair[1], text])
        logger.info(f"Successfully dumped {output_path} !")


if __name__ == "__main__":

    rinna_ja_gpt = RinnaJapaneseGpt()

    # 刺激語と連想語と抽出文数と抽出文からなるデータを読み込む
    input_dir = "datasets/連想語頻度表/pairs"
    input_path = f"{input_dir}/htns_200_best10_pairs.csv.gz"
    with gzip.open(input_path, 'rt') as f:
        reader = csv.reader(f)
        all_data = [[*row[:2], int(row[2]), eval(row[-1])] for row in reader]

    # 使用するデータをサンプリング
    sample_data = [row for row in all_data if row[2]>2][:3]
    sample_pairs = [[*row[:2]] for row in sample_data]

    # プロンプト入力用のテンプレートを読み込む
    template_dir = "datasets/連想語頻度表/templates"
    template_path = f"{template_dir}/zero-shot_no_refs.json"
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
            references = [f"{i+1}. {ref}" for i, ref in enumerate(row[-1])]
            input_text = input_text.replace("{references}", "\n".join(references))
        input_text = input_text.replace("{input_slot}", input_text)
        input_texts.append(input_text)

    encoded_texts = rinna_ja_gpt.encode_texts(input_texts)
    output_texts = rinna_ja_gpt.generate_texts(encoded_texts)

    output_dir = "results/ja/連想語頻度表/text_generation"
    rinna_ja_gpt.dump_result(output_dir, output_texts, sample_pairs)

    logger.info("All done")
