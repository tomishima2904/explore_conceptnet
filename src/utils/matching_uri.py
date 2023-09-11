import gzip
import csv
import re


def matching_uri(triplet: list, entity: str):
    graph = {}
    for r, h, t in triplet:
        if h not in graph:
            graph[h] = []
        graph[h].append((r, t))

    # もし entity を含む他のURIも欲しかったら下行を使用
    # pattern = r".*(" + re.escape(entity) + r"(\/|$))"

    pattern = r".*" + re.escape(entity) + r"$"
    matched_entities = set([])

    for key in graph.keys():
        if re.search(pattern, key):
            matched_entities.add(key)

    return matched_entities


if __name__ == "__main__":

    # 使用する ConceptNet のデータのパス
    dataset_dir = "datasets/conceptnet"
    lang = "ja"
    input_file = f"conceptnet-assertions-5.7.0_{lang}.csv.gz"
    conceptnet_path = f"{dataset_dir}/{input_file}"

    conceptnet_triplet = []
    with gzip.open(conceptnet_path, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            conceptnet_triplet.append(row[1:-1])  # 0番目と4番目をメモリ節約のため排除

    entity = "犬"
    entity = f"/c/{lang}/{entity}"

    matched_uri = matching_uri(conceptnet_triplet, entity)
    print(matched_uri)
