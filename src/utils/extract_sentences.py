import csv
import gzip
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
        print(f"{datetime.datetime.now()}: Start")
        sys.stdout.flush()

        for i, row in enumerate(conceptnet):
            head = row[1]
            tail = row[2]

            sentences = []
            for sentence in corpus:
                if head in sentence and tail in sentence:
                    sentences.append(sentence)
                    # if len(sentences) == num_extract:
                    #     break

            data = (row[0], head, tail, sentences)
            writer.writerow(data)

            if i % 100 == 0:
                sys.stdout.flush() # 明示的にflush
                print(f"{datetime.datetime.now()}: {i} triplets have been processed.")
    print(f"Successfully dumped {output_path} !")


# 抽出文がないものを除外し、文末の改行コードを除去
def clean_sentences(input_path: str, output_path: str):
    with gzip.open(input_path, 'rt') as f:
        reader = csv.reader(f)

    with gzip.open(output_path, 'wt') as f:
        writer = csv.writer(f)
        for i, row in enumerate(reader):
            # 抽出文が無い場合新しいデータセットには記述しない
            if row[-1] == "[]":
                continue
            else:
                sentences = eval(row[-1])
                cleaned_sentences = [sentence.rstrip('\n') for sentence in sentences]
                writer.writerow((*row[:-1], cleaned_sentences))
    print(f"Successfully dumped {output_path} !")


if __name__ == "__main__":
    # ConceptNetをロード
    lang = "ja"
    dataset_type = "1"
    dataset_dir = f"datasets/conceptnet-assertions-5.7.0/{lang}"
    input_file = f"triplets_{dataset_type}.csv"
    conceptnet_path = f"{dataset_dir}/{input_file}"
    conceptnet = fh.read_csv(conceptnet_path, has_header=False)

    # 日本語Wikipediaをロード
    corpus_dir = "datasets/jawiki-20221226"
    input_corpus_path = f"{corpus_dir}/jawiki_sentences_200.txt.gz"

    print("Loading wikipedia corpus ...")
    with gzip.open(input_corpus_path, "rt") as f:
        corpus = f.readlines()

    # # STAIR Captionsをロード
    # corpus_dir = "datasets/STAIR-captions"
    # input_corpus_path = f"{corpus_dir}/stair_captions_v1.2_train_sentences.txt"

    # print("Loading STAIR Captions corpus ...")
    # with open(input_corpus_path, "r") as f:
    #     corpus = f.readlines()

    output_dir = "datasets/rel_gen/origin_rhts"
    output_corpus_path = f"{output_dir}/origin_rhts_200_{dataset_type}.csv.gz"

    print("Extracting sentences ...")
    extract_sentences(output_path=output_corpus_path,
                      corpus=corpus,
                      conceptnet=conceptnet)

    print("Claning sentences ...")
    cleaned_output_dir = "datasets/rel_gen/cleaned_rhts"
    output_cleaned_path = f"{cleaned_output_dir}/cleaned_rhts_200_{dataset_type}.csv.gz"
    clean_sentences(input_path=output_corpus_path, output_path=output_cleaned_path)
