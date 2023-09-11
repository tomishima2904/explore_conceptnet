import csv


def find_matching_pairs(file_path):
    pairs = []
    data = {}

    # CSVファイルの読み込み
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)

        # データを処理して辞書に格納
        for row in reader:
            word1 = row[1]
            word2 = row[2]
            key1 = (word1, word2)
            key2 = (word2, word1)
            if key1 in data:
                if row not in data[key1]:
                    data[key1].append(row)
            elif key2 in data:
                if row not in data[key2]:
                    data[key2].append(row)
            else:
                data[key1] = [row]

        # 組み合わせが2つ以上あるペアを取得
        for key, values in data.items():
            if len(values) > 1:
                pairs.append(values)

    return pairs


# ファイルパスを指定して関数を呼び出す
dataset_dir = "datasets/conceptnet-assertions-5.7.0/ja"
input_path = f"{dataset_dir}/origin_triplets.csv"
output_path = f"{dataset_dir}/matcning_pairs.csv"
matching_pairs = find_matching_pairs(input_path)
with open(output_path, "w") as f:
    writer = csv.writer(f)
    writer.writerows(matching_pairs)
