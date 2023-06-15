import gzip
import csv

from extract_entity import extract_entity
from normalize_neologd import normalize_neologd


def conceptnet_to_triplets(input_path, output_path, removed_path, lang="ja"):
    conceptnet = []
    relations = []
    with gzip.open(input_path, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            conceptnet.append(row[1:-1])  # relation と head と tail だけ取得
            relations.append(row[1])

    # 例. relation_mapping["/r/RelatedTo"] = 関連する
    # 参照. datasets/conceptnet-assertions-5.7.0/ja/relations.csv
    relations_set = set(relations)
    all_data = {relation_uri: [] for relation_uri in relations_set}

    with gzip.open(removed_path, 'wt') as f:
        writer = csv.writer(f)
        for i, row in enumerate(conceptnet):
            relation_uri = row[0]

            # `/r/ExternalURL` はデータセットに含めない
            if relation_uri == "/r/ExternalURL":
                continue

            # `NotUsedFor`などの否定的な関係性は含めない
            if "Not" in relation_uri:
                continue

            head = extract_entity(lang, row[1])
            tail = extract_entity(lang, row[2])
            if head == None or tail == None:
                writer.writerow((i, *row[:]))
                continue

            # 正規化
            head = normalize_neologd(head)
            tail = normalize_neologd(tail)
            # 例. "イタリアン_コーヒー" -> "イタリアンコーヒー"
            if lang == "ja":
                head = head.replace("_", "")
                tail = tail.replace("_", "")
            # 例. "New_York" -> "New York"
            else:
                head = head.replace("_", " ")
                tail = tail.replace("_", " ")

            all_data[relation_uri].append([head, tail])

    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        for relation_uri, heads_and_tails in all_data.items():
            # (relation, head, tail) のトリプレットを作成
            triplets = [[relation_uri, pair[0], pair[1]] for pair in heads_and_tails]
            writer.writerows(triplets)


def make_word_pairs(input_path, output_path):
    word_pairs = set()
    with open(input_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            word_pair = tuple(sorted((row[1], row[2])))
            word_pairs.add(word_pair)

    with open(output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(word_pairs)


if __name__ == "__main__":
    lang = "ja"
    dataset_dir = f"datasets/conceptnet-assertions-5.7.0/{lang}"

    input_path = f"{dataset_dir}/conceptnet-assertions-5.7.0_{lang}.csv.gz"
    output_path = f"{dataset_dir}/origin_triplets.csv"
    removed_path = f"{dataset_dir}/removed_triplets.csv.gz"
    conceptnet_to_triplets(input_path, output_path, removed_path)

    input_path = f"{dataset_dir}/origin_triplets.csv"
    output_path = f"{dataset_dir}/origin_word_pairs.csv"
    make_word_pairs(input_path, output_path)
