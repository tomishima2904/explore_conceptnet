import gzip
import csv


# `lang` で指定した言語の entity のみからなる ConceptNet を作成
def minify_concepnet(input_path: str, output_path: str, lang: str = "ja") -> None:
    with gzip.open(input_path, 'rt') as rf:
        reader = csv.reader(rf, delimiter='\t')
        with gzip.open(output_path, 'wt') as wf:
            writer = csv.writer(wf, delimiter='\t')
            for row in reader:
                if (row[2].startswith(f"/c/{lang}/") and row[3].startswith(f"/c/{lang}/")) or \
                    (row[2].startswith(f"/c/en/") and row[3].startswith(f"/c/en/")) or \
                    (row[2].startswith(f"/c/en/") and row[3].startswith(f"/c/ja/")) or \
                    (row[2].startswith(f"/c/ja/") and row[3].startswith(f"/c/en/")):
                    writer.writerow(row)
    print(f"Successfully dumped {output_path} !")


if __name__ == "__main__":
    lang = "ja"
    dataset_dir = f"datasets/conceptnet-assertions-5.7.0"
    input_path = f"{dataset_dir}/conceptnet-assertions-5.7.0.csv.gz"
    output_path = f"{dataset_dir}/conceptnet-assertions-5.7.0_en_{lang}.csv.gz"

    minify_concepnet(input_path, output_path, lang)
