import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

import csv
import gzip
import datetime
from tqdm import tqdm
import logzero
from logzero import logger

from src.utils.file_handlers import get_8char_datetime


gb_to_bytes = 20 * 1024 * 1024 * 1024
csv.field_size_limit(gb_to_bytes)

set_seed(19990429)

logger.info("Loading model ...")
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b")

if torch.cuda.is_available():
    model = model.to("cuda")


def generate_text(encoded_texts):
    with torch.no_grad():
        output_ids_list = []
        for i, encoded_text in enumerate(encoded_texts):
            output_ids = model.generate(
                encoded_text,
                max_new_tokens=100,
                min_new_tokens=100,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=3,
            )
            output_ids_list.append(output_ids)
            logger.info(f"{i} sentences have been processed")

    # モデルからの出力をデコード
    output_texts = [[tokenizer.decode(output_sentence)
                     for output_sentence in output_sentences]
                    for output_sentences in output_ids_list]
    return output_texts


def dump_result(output_path, output_texts, sample_words):
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        for pair, text in zip(sample_words, output_texts):
            writer.writerow([pair[0], pair[1], text])


input_dir = "datasets/連想語頻度表/pairs"
input_path = f"{input_dir}/htns_200_best10_pairs.csv.gz"
with gzip.open(input_path, 'rt') as f:
    reader = csv.reader(f)
    all_data = [[*row[:2], int(row[2]), eval(row[-1])] for row in reader]

sample_pairs = all_data[:3]
only_word_pairs = [[*row[:2]] for row in sample_pairs]
input_texts = [f"「{row[0]}」と「{row[1]}」の関係性は、" for row in sample_pairs]

encoded_texts = [tokenizer.encode(text, add_special_tokens=False, return_tensors="pt") for text in input_texts]
encoded_texts = [text.to(model.device) for text in encoded_texts]
output_texts = generate_text(encoded_texts)

output_dir = "results/ja/連想語頻度表/text_generation"
date_time = get_8char_datetime()
output_path = f"{output_dir}/{date_time}.csv"
dump_result(output_path, output_texts, sample_pairs)

print("All done")
