import csv
import re
import json

def result_formatter(result_dir: str, num_refs: int, template_path: str):
    input_path = f"{result_dir}/generated_texts.csv"

    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        all_data = [row for row in reader]

    # 出力結果を見やすくするために、結果の比較に不要な部分 (replace_template) を空文字列に置換
    with open(template_path, 'r') as f:
        templates = json.load(f)
        replace_template: str = templates["replace_template"]

    # 参照文なし
    output_path = f"{result_dir}/formatted_results.txt"
    reference_text = "上記の文を参考にして、"
    if num_refs == 0:
        with open(output_path, 'w') as f:
            for row in all_data:
                head = row[0]
                tail = row[1]
                rel = eval(row[2])[0]
                generated_sentences = eval(row[-1])

                f.write(f"{rel}, {head}, {tail}\n")
                for s in generated_sentences:
                    replaced_s = s.replace(replace_template, "")
                    f.write(f"{replaced_s}\n")
                f.write("\n")
    # 参照文あり
    else:
        with open(output_path, 'w') as f:
            for row in all_data:
                head = row[0]
                tail = row[1]
                rel = row[2]
                generated_sentences = eval(row[-1])

                f.write(f"{rel}, {head}, {tail}\n")

                for i, generated_sentence in enumerate(generated_sentences):
                    replaced_sentece = generated_sentence.replace(replace_template, "")
                    reference_position = replaced_sentece.index(reference_text)
                    after_text = replaced_sentece[reference_position + len(reference_text):]
                    f.write(after_text)
                    f.write("\n")
                f.write("\n")
    print(f"Successfully dumped {output_path}")

    output_path = f"{result_dir}/formatted_results.csv"

    # 結果をCSV形式でも出力

    # 参照文なし
    if num_refs == 0:
        with open(output_path, 'w') as f:
            writer = csv.writer(f)
            for i, row in enumerate(all_data):
                head = row[0]
                tail = row[1]
                rel = eval(row[2])[0]
                generated_sentences = eval(row[-1])
                replaced_senteces = [s.replace(replace_template, "") for s in generated_sentences]
                writer.writerow([i, rel, head, tail, replaced_senteces])

    # 参照文あり
    else:
        with open(output_path, 'w') as f:
            writer = csv.writer(f)
            for j, row in enumerate(all_data):
                head = row[0]
                tail = row[1]
                rel = row[2]
                generated_sentences = eval(row[-1])

                texts = []

                for i, generated_sentence in enumerate(generated_sentences):
                    replaced_sentece = generated_sentence.replace(replace_template, "")
                    reference_position = replaced_sentece.index(reference_text)
                    after_text = replaced_sentece[reference_position + len(reference_text):]
                    texts.append(after_text)
                writer.writerow([j, rel, head, tail, texts])
    print(f"Successfully dumped {output_path}")
