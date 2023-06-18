import csv
import gzip
import datetime
import sys


# 抽出文がないものを除外し、文末の改行コードを除去
def clean_sentences(input_path: str, output_path: str, err_path: str):
    with gzip.open(input_path, 'rt') as rf, \
        gzip.open(output_path, 'wt') as wf, \
        open(err_path, 'w') as ef:
        reader = csv.reader(rf)
        writer = csv.writer(wf)
        err_writer = csv.writer(ef)
        for i, row in enumerate(reader):
            # 抽出文が無い場合新しいデータセットには記述しない
            if row[-1] == "[]":
                continue
            else:
                sentences = eval(row[-1])
                cleaned_sentences = [sentence.rstrip('\n') for sentence in sentences]
                data = [*row[:-1], cleaned_sentences]
                if len(data) == 4:
                    writer.writerow(data)
                else:
                    err_writer.writerow([i+1, *data])

            if i % 100 == 0:
                sys.stdout.flush() # 明示的にflush
                print(f"{datetime.datetime.now()}: {i} lines have been processed.")
    print(f"Successfully dumped {output_path} !")


if __name__ == "__main__":
    # 20GBまでのcsvファイルを扱えるようにフィールドサイズ制限を増やす
    gb_to_bytes = 20 * 1024 * 1024 * 1024
    csv.field_size_limit(gb_to_bytes)

    dataset_type = "1"
    input_dir = "datasets/rel_gen/origin_rhts"
    input_path = f"{input_dir}/origin_rhts_200_{dataset_type}.csv.gz"
    output_dir = "datasets/rel_gen/cleaned_rhts"
    output_path = f"{output_dir}/cleaned_rhts_200_{dataset_type}.csv.gz"
    err_path = f"{output_dir}/cleaned_rhts_200_{dataset_type}_error.csv.gz"

    print("Cleaning sentences ...")
    clean_sentences(input_path, output_path=output_path, err_path=err_path)
