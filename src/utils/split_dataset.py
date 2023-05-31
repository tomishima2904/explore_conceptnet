import gzip
import csv
from sklearn.model_selection import train_test_split
import random
import sys

import file_handlers as fh
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
relations_data, _ = fh.read_csv(relations_data_path, has_header=True)
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

train_data = []
val_data = []
test_data = []

for relation_uri, heads_and_tails in all_data.items():
    # (relation, head, tail) のトリプレットを作成
    triplets = [[relation_mapping[relation_uri], pair[0], pair[1]]
                for pair in heads_and_tails]

    # URIが`/r/RelatedTo`なら全てテストセットに回す
    if relation_uri == "/r/RelatedTo":
        test_data.extend(triplets)

    # train:val:test = 8:1:1 に分割
    else:
        if len(triplets) < 1:
            continue
        # 8:2で分割
        train, val_and_test = train_test_split(triplets,
                                               train_size=0.8,
                                               test_size=0.2,
                                               shuffle=True,
                                               random_state=19990429)

        if len(val_and_test) < 2:
            continue
        # (8):1:1で分割
        val, test = train_test_split(val_and_test,
                                     train_size=0.5,
                                     test_size=0.5,
                                     shuffle=True,
                                     random_state=19990429)
        train_data.extend(train)
        val_data.extend(val)
        test_data.extend(test)

random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

output_dir = dataset_dir
output_train_path = f"{output_dir}/train_triplets.csv"
output_val_path = f"{output_dir}/val_triplets.csv"
output_test_path = f"{output_dir}/test_triplets.csv"

fh.write_csv(path=output_train_path, data=train_data)
fh.write_csv(path=output_val_path, data=val_data)
fh.write_csv(path=output_test_path, data=test_data)
