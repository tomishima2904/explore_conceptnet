import gzip
import csv

from extract_entity import extract_entity
from normalize_neologd import normalize_neologd


lang = "ja"
dataset_dir = f"datasets/conceptnet-assertions-5.7.0/{lang}"
input_file = f"conceptnet-assertions-5.7.0_{lang}.csv.gz"
conceptnet_path = f"{dataset_dir}/{input_file}"

conceptnet = []
with gzip.open(conceptnet_path, 'rt') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        conceptnet.append(row[1:-1])  # relation と head と tail だけ取得

# 例. relation_mapping["/r/RelatedTo"] = 関連する
# 参照. datasets/conceptnet-assertions-5.7.0/ja/relations.csv
relations_data_path = f"{dataset_dir}/relations.csv"
with open(relations_data_path, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    relations_data = [row for row in reader]
relation_mapping = {row[0]: row[2] for row in relations_data}

all_data = {relation_uri: [] for relation_uri in relation_mapping.keys()}

for row in conceptnet:
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

output_dir = dataset_dir
output_path = f"{output_dir}/origin_triplets.csv"

with open(output_path, 'w') as f:
    writer = csv.writer(f)
    for relation_uri, heads_and_tails in all_data.items():
        # (relation, head, tail) のトリプレットを作成
        triplets = [[relation_mapping[relation_uri], pair[0], pair[1]]
                    for pair in heads_and_tails]
        writer.writerow(triplets)
