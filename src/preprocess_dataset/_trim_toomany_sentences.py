# 抽出文が limit (20万) 以上あるものとそうでないものを分離する
# この機能はすでに`extract_sentences.py`に組み込み済み

import gzip
import csv
import sys
import datetime


# 20GBまでのcsvファイルを扱えるようにフィールドサイズ制限を増やす
gb_to_bytes = 50 * 1024 * 1024 * 1024
csv.field_size_limit(gb_to_bytes)


def trim_toomany_sentences(input_path, output_path, trimmed_path, limit=200000):
    with gzip.open(input_path, 'rt') as rf, \
        gzip.open(output_path, 'wt') as wf, \
        gzip.open(trimmed_path, 'wt') as tf:

        reader = csv.reader(rf)
        writer = csv.writer(wf)
        t_writer = csv.writer(tf)

        for i, row in enumerate(reader):
            sentences = eval(row[-1])
            if len(sentences) < limit:
                writer.writerow(row)
            else:
                t_writer.writerow(row)

            print(f"{datetime.datetime.now()}: {i}")
            sys.stdout.flush() # 明示的にflush

    print("Done!")


if __name__ == "__main__":
    dataset_dir = "datasets/rel_gen/cleaned_rhts"
    dataset_id = "5678"
    input_path = f"{dataset_dir}/cleaned_rhts_200_{dataset_id}_all.csv.gz"
    output_path = f"{dataset_dir}/cleand_rhts_200_{dataset_id}.csv.gz"
    trimmed_path = f"{dataset_dir}/cleand_rhts_200_{dataset_id}_overlimit.csv.gz"

    print("Loading dataset ...")
    sys.stdout.flush() # 明示的にflush
    trim_toomany_sentences(input_path, output_path, trimmed_path)


