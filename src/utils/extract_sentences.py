import csv
import gzip
import json
from tqdm import tqdm

import file_handlers as fh
from extract_entity import extract_entity


def _extract_sentences_from_corpus(head: str, tail: str, corpus: list, num_extract=3) -> list:
    sentences = []
    for sentence in corpus:
        if head in sentence:
            if tail in sentence:
                sentences.append(sentence)
                if len(sentences) == num_extract:
                    break
    return sentences


def extract_sentences(
        output_path: str,
        corpus: list,
        head_and_tail: list,
        lang: str = "ja") -> None:
    # ConceptNet から unique な (head, tail) のペアを抽出
    unique_pairs = set([])
    for pair in head_and_tail:
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
            _extract_sentences_from_corpus(head_entity, tail_entity, corpus)

    with open(output_path, "w") as f:
        json.dump(sentences_dict_corpus, f, ensure_ascii=False)
    print(f"Successfully dumped {output_path} !")


if __name__ == "__main__":
    # 日本語Wikipediaをロード
    corpus_dir = "datasets/jawiki-20221226"
    input_corpus_path = f"{corpus_dir}/train_1.9M.txt.gz"

    print("Loading wikipedia corpus ...")
    with gzip.open(input_corpus_path, "rt") as f:
        corpus = f.readlines()

    # # STAIR Captionsをロード
    # corpus_dir = "datasets/STAIR-captions"
    # input_corpus_path = f"{corpus_dir}/stair_captions_v1.2_train_sentences.txt"

    # print("Loading STAIR Captions corpus ...")
    # with open(input_corpus_path, "r") as f:
    #     corpus = f.readlines()

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

    fh.makedirs(corpus_dir)
    output_corpus_path = f"{corpus_dir}/extracted_corpus.json"

    extract_sentences(output_path=output_corpus_path,
                      corpus=corpus,
                      head_and_tail=conceptnet)
