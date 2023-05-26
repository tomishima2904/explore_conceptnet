import os
import csv
import file_handlers as fh


def summarize_routes(input_dir: str, output_path: str) -> None:

    file_names = os.listdir(input_dir)

    with open(output_path, "w") as wf:
        writer = csv.writer(wf)
        writer.writerow(("source", "source_uri", "target_uri", "target" , "len", "path"))

        for file_name in file_names:
            input_file = f"{input_dir}/{file_name}"
            head_entity = file_name.replace('.csv', '')
            df = fh.read_csv_as_df(input_file, header=0)

            for row in df.itertuples():
                writer.writerow((head_entity, *row))

    print(f"Successfully dumped {output_path} !")


if __name__ == "__main__":
    result_dir = "results/ja/連想語頻度表"
    char_type = "カタカナ"
    input_dir = f"{result_dir}/{char_type}"
    fh.makedirs(input_dir)

    output_path = f"{result_dir}/{char_type}.csv"
    summarize_routes(input_dir, output_path)
