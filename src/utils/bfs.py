from collections import deque
import gzip
import csv
import re
import json
import sys
import logzero
from logzero import logger

import file_handlers as fh
from summarize_routes import summarize_routes


def find_shortest_path(data: list, source: str, target: str) -> tuple:
    graph = {}
    for r, h, t in data:
        if h not in graph:
            graph[h] = []
        graph[h].append((r, t))

    re_start = r".*" + re.escape(source) + r"$"
    re_end = r".*" + re.escape(target) + r"$"

    start_and_end_entity = [None, None]

    # entity の 正確な URIを見つける
    for key in graph.keys():
        if None in start_and_end_entity:
            if re.search(re_start, key):
                start = key
                start_and_end_entity[0] = key
            if re.search(re_end, key):
                end = key
                start_and_end_entity[1] = key
        else:
            break

    # start or end の uri が存在しない場合
    # TODO: 名寄せ問題
    if None in start_and_end_entity:
        logger.info(f"Entity doesn't exist in data")
        return start_and_end_entity, [([], [])]

    # 幅優先探索 (bfs)
    queue = deque([(start, [], [])])
    visited = set()
    shortest_paths = []
    min_path_length = float('inf')

    while queue:
        node, path, relations = queue.popleft()

        if len(path) > min_path_length:
            # より短い経路が見つかっている場合、探索終了
            logger.info(shortest_paths)
            break

        # node が target の URI の条件に合致したら
        # すなわち、最短経路が見つかった場合、経路と共に返す
        if node == end:
            # より短い経路が見つかった場合、それまでの経路を無効化
            if len(path) < min_path_length:
                shortest_paths = []
                min_path_length = len(path)

            # 現在の経路と関係を最短経路として追加
            shortest_paths.append((path + [node], relations))
            continue

        if node in graph:
            for r, neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [node], relations + [r]))
                    visited.add(neighbor)

    # 経路がない場合
    if shortest_paths == []:
        logger.info(f"No paths found from {source} to {target}.")
        shortest_paths = [([], [])]

    return start_and_end_entity, shortest_paths


if __name__ == "__main__":

    lang = "en_ja"
    dataset_dir = f"datasets/conceptnet-assertions-5.7.0/{lang}"
    input_file = f"conceptnet-assertions-5.7.0_{lang}.csv.gz"
    conceptnet_path = f"{dataset_dir}/{input_file}"

    conceptnet = []
    with gzip.open(conceptnet_path, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            conceptnet.append(row[1:-1])  # 0番目と4番目をメモリ節約のため排除

    """conceptnet
    以下の5つの要素から構成
    0. The URI of the whole edge.  例. /a/[/r/Antonym/,/c/ja/mt/,/c/ja/オートマチック/]
    1. The relation expressed by the edge.  例. /r/Antonym
    2. The node at the start of the edge. 例. /c/ja/mt
    3. The node at the end of the edge.  例. /c/ja/オートマチック
    4. A JSON structure of additional information about the edge, such as its weight
    例. "{""dataset"": ""/d/wiktionary/en"", ""license"": ""cc:by-sa/4.0"", ""sources"": [{""contributor"": ""/s/resource/wiktionary/en"", ""process"": ""/s/process/wikiparsec/2""}], ""weight"": 1.0}"
    """

    # head, tail, 連想強度
    input_path = "datasets/連想語頻度表/pairs/all_htp.csv"

    output_dir = "datasets/連想語頻度表/pairs"
    fh.makedirs(output_dir)
    output_path = f"{output_dir}/all_routes2.csv"

    datetime = fh.get_12chars_datetime()
    logzero.logfile(f"logs/{datetime}.log")
    sys.stderr = logzero.setup_logger(formatter=logzero.LogFormatter())  # エラー出力をLogzeroのログにリダイレクト

    with open(input_path, 'r') as rf, open(output_path, 'w') as wf:
        reader = csv.reader(rf)
        header = next(reader)
        writer = csv.writer(wf)

        # 出力ファイルのheader
        # writer.writerow(("source", "source_uri", "target", "target_uri", "hops", "paths", "relations"))

        for i, row in enumerate(reader):
            # 一つの刺激語に対して最大10個の連想語へのルートを幅優先探索
            if int(row[1]) > 10:
                continue

            head_entity = row[2]
            tail_entity = row[3]

            source_uri = f"/c/ja/{head_entity}"
            target_uri = f"/c/ja/{tail_entity}"

            start_and_end, shortest_paths_and_rels = find_shortest_path(conceptnet, source_uri, target_uri)
            source_uri, target_uri = start_and_end
            shortest_paths = [pair[0] for pair in shortest_paths_and_rels]
            relations = [pair[1] for pair in shortest_paths_and_rels]
            shortest_path_len = len(relations[0])
            """
            start_and_end: (source_uri, target_uri). ConceptNet上に存在しない場合はNone
            shortest_paths_and_rels: (shortest_paths, relations)
            shortest_pahts: source_uri から target_uri までの最短経路長にあるノード (uri)
            relations: ノード間を繋ぐ relation type
            """

            writer.writerow((head_entity, source_uri, tail_entity, target_uri, shortest_path_len, shortest_paths, relations))
            logger.info(f"{i}: {head_entity} & {tail_entity} done")

    logger.info("All done!")

    # result_dir = f"results/{lang}/連想語頻度表"
    # input_dir = f"{result_dir}/{char_type}"
    # output_path = f"{result_dir}/{char_type}.csv"
    # summarize_routes(input_dir, output_path)
