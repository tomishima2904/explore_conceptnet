import csv
import gzip
import datetime
import sys


def extract_sentences(
        output_path: str,
        over1m_path: str,
        corpus: list,
        conceptnet: list,
        num_extract: int = 5) -> None:
    with gzip.open(output_path, "wt") as f, gzip.open(over1m_path, "wt") as f1m:
        writer = csv.writer(f)
        writer_over1m = csv.writer(f1m)
        print(f"{datetime.datetime.now()}: Start")
        sys.stdout.flush()

        word_pairs = set()

        for i, row in enumerate(conceptnet):
            head = row[1]
            tail = row[2]
            word_pair = tuple(sorted((head, tail)))

            if word_pair not in word_pairs:
                word_pairs.add(word_pair)
                sentences = []
                for sentence in corpus:
                    if head in sentence and tail in sentence:
                        sentences.append(sentence)
                        # if len(sentences) == num_extract:
                        #     break

                num_sentences = len(sentences)
                data = (head, tail, num_sentences, sentences)
                if num_sentences < 1000000:
                    writer.writerow(data)
                else:
                    writer_over1m.writerow(data)

            if i % 100 == 0:
                sys.stdout.flush() # 明示的にflush
                print(f"{datetime.datetime.now()}: {i} triplets have been processed.")
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
    output_over1m_path = f"{output_dir}/origin_htns_200_{dataset_type}_over1m.csv.gz"

    print("Extracting sentences ...")
    extract_sentences(output_path=output_corpus_path,
                      over1m_path=output_over1m_path,
                      corpus=corpus,
                      conceptnet=conceptnet)

