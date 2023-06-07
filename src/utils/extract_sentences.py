import csv
import gzip
import random
import datetime
import sys

import file_handlers as fh


def extract_sentences(
        output_path: str,
        corpus: list,
        conceptnet: list,
        num_extract: int = 5) -> None:
    with gzip.open(output_path, "wt") as f:
        writer = csv.writer(f)
        i = 0
        print(f"{datetime.datetime.now()}: Start")
        sys.stdout.flush()
        for row in conceptnet:
            # URI から entity を抽出
            head = row[1]
            tail = row[2]

            sentences = []
            for sentence in corpus:
                if head in sentence and tail in sentence:
                    sentences.append(sentence)
                    if len(sentences) == num_extract:
                        break

            data = (row[0], head, tail, sentences)
            writer.writerow(data)

            i += 1
            if i % 100 == 0:
                sys.stdout.flush() # 明示的にflush
                print(f"{datetime.datetime.now()}: {i} triplets have been processed.")
    print(f"Successfully dumped {output_path} !")


if __name__ == "__main__":
    # ConceptNetをロード
    lang = "ja"
    dataset_type = "train"
    dataset_dir = f"datasets/conceptnet-assertions-5.7.0/{lang}"
    input_file = f"{dataset_type}_triplets.csv"
    conceptnet_path = f"{dataset_dir}/{input_file}"
    conceptnet = fh.read_csv(conceptnet_path, has_header=False)

    # 日本語Wikipediaをロード
    corpus_dir = "datasets/jawiki-20221226"
    input_corpus_path = f"{corpus_dir}/jawiki_sentences_1000.txt.gz"

    print("Loading wikipedia corpus ...")
    with gzip.open(input_corpus_path, "rt") as f:
        corpus = f.readlines()
    random.shuffle(corpus)

    # # STAIR Captionsをロード
    # corpus_dir = "datasets/STAIR-captions"
    # input_corpus_path = f"{corpus_dir}/stair_captions_v1.2_train_sentences.txt"

    # print("Loading STAIR Captions corpus ...")
    # with open(input_corpus_path, "r") as f:
    #     corpus = f.readlines()

    output_corpus_path = f"{corpus_dir}/origin_rhts_{dataset_type}.csv.gz"

    print("Extracting sentences ...")
    extract_sentences(output_path=output_corpus_path,
                      corpus=corpus,
                      conceptnet=conceptnet)
