import csv
import gzip
import datetime
import sys


def extract_sentences(
        output_path: str,
        over_path: str,
        corpus: list,
        conceptnet: list,
        extract_limit: int = 200000) -> None:
    with gzip.open(output_path, "wt") as f, gzip.open(over_path, "wt") as f_over:
        writer = csv.writer(f)
        writer_over = csv.writer(f_over)
        print(f"{datetime.datetime.now()}: Start")
        sys.stdout.flush()

        for i, row in enumerate(conceptnet):
            head = row[1]
            tail = row[2]

            sentences = []
            for sentence in corpus:
                if head in sentence and tail in sentence:
                    sentences.append(sentence)
                    if len(sentences) == extract_limit:
                        break

            num_sentences = len(sentences)
            data = (head, tail, num_sentences, sentences)

            # 抽出文の総数があまりにも多い場合は別のファイルへ出力
            if num_sentences < extract_limit:
                writer.writerow(data)
            else:
                writer_over.writerow(data)

            if i % 100 == 0:
                print(f"{datetime.datetime.now()}: {i} triplets have been processed.")
                sys.stdout.flush() # 明示的にflush
    print(f"Successfully dumped {output_path} !")


if __name__ == "__main__":
    # ConceptNetをロード
    lang = "ja"
    dataset_type = "1"
    dataset_dir = f"datasets/conceptnet-assertions-5.7.0/{lang}"
    input_file = f"triplets_{dataset_type}.csv"
    conceptnet_path = f"{dataset_dir}/{input_file}"
    with open(conceptnet_path, 'r') as f:
        reader = csv.reader(f)
        conceptnet = [row for row in reader]

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
    output_corpus_path = f"{output_dir}/origin_htns_200_{dataset_type}.csv.gz"
    output_over_path = f"{output_dir}/origin_htns_200_{dataset_type}_over.csv.gz"

    print("Extracting sentences ...")
    extract_sentences(output_path=output_corpus_path,
                      over_path=output_over_path,
                      corpus=corpus,
                      conceptnet=conceptnet)

