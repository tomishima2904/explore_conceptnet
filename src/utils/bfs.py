from collections import deque
import gzip
import csv
import re
import json
import sys
import os
import datetime
import logzero
from logzero import logger
from collections import defaultdict, deque

# from summarize_routes import summarize_routes


def find_shortest_paths(data: list, start_uri: str, end_uri: str) -> tuple:
    # グラフを隣接リスト形式で表現
    adjacency_list = defaultdict(list)
    edge_relations = defaultdict(list)
    for r, h, t in data:
        adjacency_list[h].append(t)
        edge_relations[(h, t)].append(r)

    # start or end の uri が ConceptNet内に存在するかを検証
    # FIXME: head にしか注目していない (おそらく問題はないが)
    re_start = r".*" + re.escape(start_uri) + r"$"
    re_end = r".*" + re.escape(end_uri) + r"$"
    start_and_end_uri = [None, None]
    for uri in adjacency_list:
        if None in start_and_end_uri:
            if re.search(re_start, uri):
                start_and_end_uri[0] = uri
            if re.search(re_end, uri):
                start_and_end_uri[1] = uri
        else:
            break

    # uri が ConceptNet上に存在しない場合
    if None in start_and_end_uri:
        logger.info(f"{start_uri} or {end_uri} don't exist in adjacency_list")
        return start_and_end_uri, [([], [])]

    # 幅優先探索で最短経路を求める
    start, end = start_and_end_uri
    queue = deque([(start, [start], [])])
    shortest_paths = []
    visited = set()
    
    while queue:
        node, path, relations = queue.popleft()
        visited.add(node)
        
        for neighbor in adjacency_list[node]:
            relation = edge_relations[(node, neighbor)]
            if neighbor not in visited:
                if neighbor == end:
                    shortest_paths.append((path + [neighbor], relations + [relation]))
                else:
                    queue.append((neighbor, path + [neighbor], relations + [relation]))

    if shortest_paths == []:
        logger.info(f"No paths found from {start} to {end}")
    else:
        logger.info(shortest_paths)

    return start_and_end_uri, shortest_paths


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
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_path = f"{output_dir}/all_routes3.csv"

    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    date_time = now.strftime('%y%m%d%H%M%S')
    logzero.logfile(f"logs/{date_time}.log")
    sys.stderr = logzero.setup_logger(formatter=logzero.LogFormatter())  # エラー出力をLogzeroのログにリダイレクト

    with open(input_path, 'r') as rf, open(output_path, 'w') as wf:
        reader = csv.reader(rf)
        header = next(reader)
        writer = csv.writer(wf)

        # 出力ファイルのheader
        writer.writerow(("source", "source_uri", "target", "target_uri", "hops", "paths", "relations"))

        for i, row in enumerate(reader):
            # 一つの刺激語に対して最大10個の連想語へのルートを幅優先探索
            if int(row[1]) > 10:
                continue

            head_entity = row[2]
            tail_entity = row[3]

            source_uri = f"/c/ja/{head_entity}"
            target_uri = f"/c/ja/{tail_entity}"

            start_and_end, shortest_paths_and_rels = find_shortest_paths(conceptnet, source_uri, target_uri)
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
