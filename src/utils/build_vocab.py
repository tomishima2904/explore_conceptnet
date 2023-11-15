import gzip
import csv
import file_handlers as fh
from preprocess_dataset.extract_entity import extract_entity


def build_vocab(input_path: str, output_path: str):
    vocab_set = set([])
    with gzip.open(input_path, 'rt') as rf:
        reader = csv.reader(rf, delimiter='\t')
        for row in reader:
            vocab_set.add(row[2])  # entity の URI

    vocab_dict = {value: i for i, value in enumerate(vocab_set)}
    fh.write_json(output_path, vocab_dict)


# uri から entity を抜き出して vocab を作成
def build_cleaned_vocab(input_path: str, output_path: str):
    vocab_set = set([])
    with gzip.open(input_path, 'rt') as rf:
        reader = csv.reader(rf, delimiter='\t')
        for row in reader:
            # URI から entity 部分のみを抽出
            cleaned_entity = extract_entity(row[2])
            vocab_set.add(cleaned_entity)
    vocab_dict = {value: i for i, value in enumerate(vocab_set)}
    fh.write_json(output_path, vocab_dict)


if __name__ == "__main__":
    lang = "ja"
    dataset_dir = f"datasets/conceptnet-assertions-5.7.0/{lang}"
    input_file = f"conceptnet-assertions-5.7.0_{lang}.csv.gz"
    input_path = f"{dataset_dir}/{input_file}"

    output_dir = f"{dataset_dir}"
    fh.makedirs(output_dir)
    output_path = f"{output_dir}/vocab_stoi.json"
    output_cleaned_path = f"{output_dir}/cleaned_vocab_stoi.json"

    # build_vocab(input_path, output_path)
    build_cleaned_vocab(input_path, output_cleaned_path, lang="ja")

