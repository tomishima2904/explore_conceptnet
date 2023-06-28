import csv
import gzip
import datetime
import sys
import os
import time
import logzero
from logzero import logger


tokenizer_type = "jumanpp"
dic = ""
assert tokenizer_type in ["jumanpp", "mecab"]
if tokenizer_type == "mecab":
    assert dic in ["ipadic", "neologd"]
    if dic == "ipadic":
        import MeCab
        tokenizer = MeCab.Tagger("-d /usr/local/lib/mecab/dic/ipadic")
    elif dic == "neologd":
        import MeCab
        tokenizer = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
elif tokenizer_type == "jumanpp":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("nlp-waseda/roberta-base-japanese-with-auto-jumanpp")


def tokenize_mecab(text:str) -> list:
    result = tokenizer.parse(text)

    # 分かち書きされた文を抽出
    lines = result.split('\n')
    words = [line.split('\t')[0] for line in lines[:-2]]

    return words


def tokenize_juman(text:str) -> list:
    return tokenizer.tokenize(text)


# 全てのターゲットトークンが対象の文に含まれていればTrueを返す
def _is_all_targets_in_sentence(targets: list, sentence: list) -> bool:
    return all(target in sentence for target in targets)


def tokenizing_filter(input_path: str, output_path: str, removed_path: str, tokenizer_type="jumanpp"):
    with gzip.open(output_path, 'wt') as wf, gzip.open(removed_path, 'wt') as removed_f:
        writer = csv.writer(wf)
        removed_writer = csv.writer(removed_f)
        with gzip.open(input_path, 'rt') as rf:
            reader = csv.reader(rf)

            for i, row in enumerate(reader):
                sentences = eval(row[-1])
                if tokenizer_type == "jumanpp":
                    tokenized_word1 = tokenize_juman(row[0])
                    tokenized_word2 = tokenize_juman(row[1])
                    tokenized_sentences = map(tokenize_juman, sentences)
                elif tokenizer_type == ["mecab"]:
                    tokenized_word1 = tokenize_mecab(row[0])
                    tokenized_word2 = tokenize_mecab(row[1])
                    tokenized_sentences = map(tokenize_mecab, sentences)

                search_target_tokens = [*tokenized_word1, *tokenized_word2]
                filtered_sentences = []
                removed_sentences = []

                for j, tokenized_sentence in enumerate(tokenized_sentences):
                    if _is_all_targets_in_sentence(search_target_tokens, tokenized_sentence):
                        filtered_sentences.append(sentences[j])
                    else:
                        removed_sentences.append(sentences[j])
                    if j % 1000 == 999:
                        logger.info(f"{i} {j}/{row[2]} {row[0]} {row[1]} +")

                writer.writerow([row[0], row[1], len(filtered_sentences), filtered_sentences])
                removed_writer.writerow([row[0], row[1], len(removed_sentences), removed_sentences])

                logger.info(f"{i} {row[0]} {row[1]} all done")

    logger.info(f"Successfully dumped {output_path} !")


if __name__ == "__main__":
    # 20GBまでのcsvファイルを扱えるようにフィールドサイズ制限を増やす
    gb_to_bytes = 20 * 1024 * 1024 * 1024
    csv.field_size_limit(gb_to_bytes)

    dataset_type = "1"
    input_dir = "datasets/rel_gen/cleaned_rhts"
    input_path = f"{input_dir}/cleaned_htns_200_{dataset_type}_ascending.csv.gz"
    output_dir = f"datasets/rel_gen/{tokenizer_type}_htns"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_path = f"{output_dir}/filtered_htns_200_{dataset_type}_1.csv.gz"
    removed_path = f"{output_dir}/removed_htns_200_{dataset_type}_1.csv.gz"

    # loggingを設定
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    date_time = now.strftime('%y%m%d%H%M%S')
    log_path = f"logs/{date_time}_{tokenizer_type}_{dataset_type}_1.log"
    logzero.logfile(log_path)

    logger.info(f"Filtering sentences ...")
    tokenizing_filter(input_path, output_path, removed_path, tokenizer_type="jumanpp")
