import gzip
import csv
import json

from extract_entity import extract_entity
from normalize_neologd import normalize_neologd


def conceptnet_to_triplets(input_path, output_path, removed_path):
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

            lang1, head = extract_entity(row[1])
            lang2, tail = extract_entity(row[2])
            if head == None or tail == None:
                writer.writerow((i, *row[:]))
                continue

            # 正規化
            head = normalize_neologd(head)
            tail = normalize_neologd(tail)
            # 例. "イタリアン_コーヒー" -> "イタリアンコーヒー"
            if lang1 == "ja":
                head = head.replace("_", "")
            # 例. "New_York" -> "New York"
            else:
                head = head.replace("_", " ")
            if lang2 == "ja":
                tail = tail.replace("_", "")
            else:
                tail = tail.replace("_", " ")

            all_data[relation_uri].append([head, tail])
            if i % 1000 == 0:
                print(f"{i} lines have been processed")

    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        for relation_uri, heads_and_tails in all_data.items():
            # (relation, head, tail) のトリプレットを作成
            triplets = [[relation_uri, pair[0], pair[1]] for pair in heads_and_tails]
            writer.writerows(triplets)


# トリプレットから単語の組を作成して出力する
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
    print(f"Successfully dumped {output_path}!")


# トリプレットからheadとtailをキーにrelationをバリューにもつ辞書 (h2t2rels.json) を作成
def make_h2t2rels(input_path, output_path):
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        conceptnet_dict = {}
        for r, h, t in reader:
            if h not in conceptnet_dict:
                conceptnet_dict[h] = {}
            if t not in conceptnet_dict[h]:
                conceptnet_dict[h][t] = []
            if r not in conceptnet_dict[h][t]:
                conceptnet_dict[h][t].append(r)

    with open(output_path, 'w') as f:
        json.dump(conceptnet_dict, f, ensure_ascii=False)
    print(f"Successfully dumped {output_path}!")


if __name__ == "__main__":
    lang = "en_ja"
    dataset_dir = f"datasets/conceptnet-assertions-5.7.0/{lang}"

    # ConceptNetにある relation_type, head, tail のトリプレットをcsv形式でdump
    input_path = f"{dataset_dir}/conceptnet-assertions-5.7.0_{lang}.csv.gz"
    output_path = f"{dataset_dir}/origin_triplets.csv"
    removed_path = f"{dataset_dir}/removed_triplets.csv.gz"
    conceptnet_to_triplets(input_path, output_path, removed_path)

    # uriからentityを抽出しているため、1つのheadとtailの組に対して複数のrelation_typeを持つものもある
    # そこで、重複を取り除いてheadとtailの組がを出力 (順序考慮)
    input_path = f"{dataset_dir}/origin_triplets.csv"
    output_path = f"{dataset_dir}/origin_word_pairs.csv"
    make_word_pairs(input_path, output_path)

    # head と tail をキーに relation をバリューにもつ辞書を作成
    input_path = f"{dataset_dir}/origin_triplets.csv"
    output_path = f"{dataset_dir}/h2t2rels.json"
    make_h2t2rels(input_path, output_path)
