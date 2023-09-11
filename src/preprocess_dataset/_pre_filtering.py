# cleanされたファイルを統合して、語の組み合わせに関して重複を許さないようにした後、
# [h, t, n, s]の全ての組を4つのファイルに分割して出力する

import gzip
import csv
import sys
import datetime


# 50GBまでのcsvファイルを扱えるようにフィールドサイズ制限を増やす
gb_to_bytes = 80 * 1024 * 1024 * 1024
csv.field_size_limit(gb_to_bytes)

dataset_dir = "datasets/rel_gen/cleaned_rhts"
input_paths = [f"{dataset_dir}/cleaned_rhts_200_{i+1}_t.csv.gz" for i in range(4)]
input_paths.append(f"{dataset_dir}/cleaned_rhts_200_5678.csv.gz")
output_path = f"{dataset_dir}/cleaned_htns_200.csv.gz"

# over_1m_path = f"{dataset_dir}/cleaned_htns_200_over1m.csv.gz"

all_data = []
all_meta_data = []
over1m_meta_data = []

word_pairs_path = f"{dataset_dir}/word_pairs_list.csv"
with open(word_pairs_path, 'r') as f:
    reader = csv.reader(f)
    word_pairs = set([tuple(row) for row in reader])

with gzip.open(output_path, 'at') as wf, open(word_pairs_path, 'w') as word_f:
    writer = csv.writer(wf)
    writer_word_pairs = csv.writer(word_f)
    writer_word_pairs.writerows(word_pairs)
    for p, input_path in enumerate(input_paths):
        with gzip.open(input_path, 'rt') as rf:
            reader = csv.reader(rf)
            for i, row in enumerate(reader):
                try:
                    word_pair = tuple(sorted((row[1], row[2])))
                    if word_pair not in word_pairs:
                        word_pairs.add(word_pair)
                        writer_word_pairs.writerow(word_pair)
                        num_sentences = len(eval(row[-1]))

                        # all_data.append([row[1], row[2], num_sentences, row[-1]])
                        writer.writerow([row[1], row[2], num_sentences, row[-1]])
                        all_meta_data.append([row[1], row[2], num_sentences])
                except Exception as e:
                    print(e)

                if i % 100 == 0:
                    print(f"{datetime.datetime.now()}: {i} lines have been processed @{input_path}")
                    sys.stdout.flush() # 明示的にflush

print("Sorting...")
all_meta_data.sort(key=lambda x:x[-1], reverse=True)

output_meta_path = f"{dataset_dir}/word_pairs_with_num_sentences.csv"

with open(output_meta_path, "w") as f:
    writer = csv.writer(f)
    writer.writerows(all_meta_data)
print("All done")

