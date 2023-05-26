from collections import deque
import gzip
import csv
import re
import os

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
        print(f"Entity doesn't exist in data")
        return start_and_end_entity, []

    # 幅優先探索 (bfs)
    queue = deque([(start, [])])
    visited = set()
    shortest_paths = []
    min_path_length = float('inf')

    while queue:
        node, path = queue.popleft()

        if len(path) > min_path_length:
            # より短い経路が見つかっている場合、探索終了
            print(shortest_paths)
            break

        # node が target の URI の条件に合致したら
        # すなわち、最短経路が見つかった場合、経路と共に返す
        if node == end:
            # より短い経路が見つかった場合、それまでの経路を無効化
            if len(path) < min_path_length:
                shortest_paths = []
                min_path_length = len(path)

            # 現在の経路を最短経路として追加
            shortest_paths.append(path + [node])

        if node in graph:
            for r, neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [node]))
                    visited.add(neighbor)

    # 経路がない場合
    if shortest_paths == []:
        print(f"No paths found from {source} to {target}.")
        shortest_paths = [[]]

    return start_and_end_entity, shortest_paths


if __name__ == "__main__":

    dataset_dir = "datasets/conceptnet"
    lang = "ja"
    input_file = f"conceptnet-assertions-5.7.0_{lang}.csv.gz"
    conceptnet_path = f"{dataset_dir}/{input_file}"

    char_type = "カタカナ"
    eval_dir = f"連想語頻度表/{char_type}"
    frequencies_dir = f"datasets/{eval_dir}"

    result_dir = f"results/{lang}"
    output_dir = f"{result_dir}/{eval_dir}"
    fh.makedirs(output_dir)

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

    # ディレクトリ内のファイル名を取得
    file_names = os.listdir(frequencies_dir)

    for file_name in file_names:
        eval_input_file = f"{frequencies_dir}/{file_name}"
        df = fh.read_csv_as_df(eval_input_file, header=0)
        head_entity = file_name.replace('.csv', '')
        start_node = f"/c/{lang}/{head_entity}"

        eval_output_file = f"{output_dir}/{file_name}"
        with open(eval_output_file, "w") as wf:
            writer = csv.writer(wf)

            # 出力ファイルのheader
            writer.writerow(("source_uri", "target_uri" "target", "len", "path"))

            for row in df.itertuples():
                tail_entity = row[2]
                target_node = f"/c/{lang}/{tail_entity}"
                start_and_end, shortest_path = find_shortest_path(conceptnet, start_node, target_node)

                shortest_path_len = len(shortest_path) - 1

                writer.writerow((*start_and_end, tail_entity, shortest_path_len, shortest_path))

                # 1つの head entity に対して 5つの tail entity へのパスを探す
                if row[0] == 5: break

    result_dir = f"results/{lang}/連想語頻度表"
    input_dir = f"{result_dir}/{char_type}"
    output_path = f"{result_dir}/{char_type}.csv"
    summarize_routes(input_dir, output_path)

