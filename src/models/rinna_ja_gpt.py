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

# シード値を固定
set_seed(19990429)

# ロギングの設定
date_time = get_8char_datetime()
log_path = f"logs/{date_time}.log"
logzero.logfile(log_path)

# モトークナイザーおよびモデルの読み込み
logger.info("Loading model ...")
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b")
model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")


def generate_text(encoded_texts: list) -> list:
    """ GPTでテキスト生成
    encoded_texts: [torch.Tensor] -> [str]
    """
    with torch.no_grad():
        output_ids_list = []
        for i, encoded_text in enumerate(encoded_texts):
            output_ids = model.generate(
                encoded_text.to(model.device),
                max_new_tokens=100,
                min_new_tokens=5,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=3,
            )
            output_ids_list.append(output_ids)
            logger.info(f"{i+1}/{len(encoded_texts)}")

    # モデルからの出力をデコード
    output_texts = []
    for output_sentences in output_ids_list:
        output_texts_per_sequence = []
        for output_sentence in output_sentences:
            # [PAD]トークンを除外
            filtered_output = [token for token in output_sentence if token != tokenizer.pad_token_id]
            decoded_output = tokenizer.decode(filtered_output)
            output_texts_per_sequence.append(decoded_output)
        output_texts.append(output_texts_per_sequence)
    return output_texts


def dump_result(output_path, output_texts, sample_words):
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        for pair, text in zip(sample_words, output_texts):
            writer.writerow([pair[0], pair[1], text])
    logger.info(f"Successfully dumped {output_path} !")


input_dir = "datasets/連想語頻度表/pairs"
input_path = f"{input_dir}/htns_200_best10_pairs.csv.gz"
with gzip.open(input_path, 'rt') as f:
    reader = csv.reader(f)
    all_data = [[*row[:2], int(row[2]), eval(row[-1])] for row in reader]

sample_pairs = [row for row in all_data if row[2]>2]
only_word_pairs = [[*row[:2]] for row in sample_pairs]

template_dir = "datasets/連想語頻度表/templates"
template_path = f"{template_dir}/hte2.txt"
logger.info(f"Template: {template_path}")
with open(template_path, "r", encoding="utf-8") as f:
    template = f.read()
input_texts = [f"{template}「{row[0]}」、「{row[1]}」:" for row in sample_pairs]

encoded_texts = [tokenizer.encode(text, add_special_tokens=False, return_tensors="pt") for text in input_texts]
output_texts = generate_text(encoded_texts)

output_dir = "results/ja/連想語頻度表/text_generation"
output_path = f"{output_dir}/{date_time}.csv"
dump_result(output_path, output_texts, sample_pairs)

logger.info("All done")
