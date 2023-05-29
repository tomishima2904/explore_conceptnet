import csv
import gzip
import json
from tqdm import tqdm

import file_handlers as fh
from extract_entity import extract_entity
from extract_sentences import extract_sentences


# ConceptNet 上にある全ての (head, tail) の組に対して、
# これら2つのentityを両方とも含む文を、日本語wikipedia内から抽出する


if __name__ == "__main__":
    corpus_dir = "datasets/jawiki-20221226"
    input_corpus_path = f"{corpus_dir}/train_500k.txt.gz"

    # 日本語Wikipediaをロード
    print("Loading wikipedia corpus ...")
    with gzip.open(input_corpus_path, "rt") as f:
        corpus = f.readlines()

    # ConceptNetをロード
    lang = "ja"
    dataset_dir = f"datasets/conceptnet-assertions-5.7.0/{lang}"
    input_file = f"conceptnet-assertions-5.7.0_{lang}.csv.gz"
    conceptnet_path = f"{dataset_dir}/{input_file}"

    conceptnet = []
    with gzip.open(conceptnet_path, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            conceptnet.append(row[2:-1])  # head と tail だけ取得

    # ConceptNet から unique な (head, tail) のペアを抽出
    unique_pairs = set([])
    for pair in conceptnet:
        # URI から entity を抽出
        head_entity = extract_entity(lang, pair[0])
        tail_entity = extract_entity(lang, pair[1])

        if head_entity == None or tail_entity == None:
            continue

        # 例. "イタリアン_コーヒー" -> "イタリアンコーヒー"
        if lang == "ja":
            head_entity = head_entity.replace("_", "")
            tail_entity = tail_entity.replace("_", "")
        # 例. "New_York" -> "New York"
        else:
            head_entity = head_entity.replace("_", " ")
            tail_entity = tail_entity.replace("_", " ")
        unique_pairs.add((head_entity, tail_entity))

    sentences_dict_corpus = {}

    for pair in tqdm(unique_pairs, total=len(unique_pairs)):
        head_entity = pair[0]
        tail_entity = pair[1]

        # head entity が キーになければ追加
        if head_entity not in sentences_dict_corpus:
            sentences_dict_corpus[head_entity] = {}

        # tail entity が キーになければ追加して、文をコーパスから抽出
        if not tail_entity in sentences_dict_corpus[head_entity]:
            sentences_dict_corpus[head_entity][tail_entity] = \
            extract_sentences(head_entity, tail_entity, corpus)

    # jsonにコーパスを出力
    output_dir = "datasets/jawiki-20221226"
    fh.makedirs(output_dir)
    output_corpus_path = f"{output_dir}/extracted_corpus.json"
    with open(output_corpus_path, "w") as f:
        json.dump(sentences_dict_corpus, f, ensure_ascii=False)
