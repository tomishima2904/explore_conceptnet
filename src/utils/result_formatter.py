import csv
import re
import json


def completion_formatter(sentence: str):
    sentence = sentence.lstrip('\n')  # 先頭の改行コードを削除
    sentence = re.sub(r'[．.。\n].*', '', sentence)  # 句点以降を削除
    return sentence


def result_formatter(input_path: str,
                     output_path_txt: str,
                     output_path_csv: str,
                     num_refs: int,
                     template_path: str,
                     model=None):

    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        all_data = [row for row in reader]

    # 出力結果を見やすくするために、結果の比較に不要な部分 (replace_template) を空文字列に置換
    with open(template_path, 'r') as f:
        template = json.load(f)
        if model == "rinna/japanese-gpt-neox-3.6b":
            replace_template: str = template["replace_template"]
        else:
            replace_template = template["prompt_input"]

    # 参照文なし
    reference_text = "上記の文を参考にして、"
    if num_refs == 0:
        with open(output_path_txt, 'w') as f:
            for row in all_data:
                head = row[0]
                tail = row[1]
                rel = eval(row[2])[0]
                generated_sentences = eval(row[-1])

                f.write(f"{head}, {tail}\n")
                for s in generated_sentences:
                    tmp_replace_template = replace_template
                    if "{head}" in tmp_replace_template:
                        tmp_replace_template = tmp_replace_template.replace("{head}", head)
                    if "{tail}" in tmp_replace_template:
                        tmp_replace_template = tmp_replace_template.replace("{tail}", tail)
                    replaced_s = s.replace(tmp_replace_template, "")
                    trimmed_s = completion_formatter(replaced_s)
                    f.write(f",{trimmed_s}\n")
                f.write("\n")
    # 参照文あり
    else:
        with open(output_path_txt, 'w') as f:
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
    print(f"Successfully dumped {output_path_txt}")

    # 結果をCSV形式でも出力

    # 参照文なし
    if num_refs == 0:
        with open(output_path_csv, 'w') as f:
            writer = csv.writer(f)
            for i, row in enumerate(all_data):
                head = row[0]
                tail = row[1]
                rel = eval(row[2])[0]
                generated_sentences = eval(row[-1])
                formatted_completions = []
                for s in generated_sentences:
                    tmp_replace_template = replace_template
                    if "{head}" in tmp_replace_template:
                        tmp_replace_template = tmp_replace_template.replace("{head}", head)
                    if "{tail}" in tmp_replace_template:
                        tmp_replace_template = tmp_replace_template.replace("{tail}", tail)
                    replaced_s = s.replace(tmp_replace_template, "")
                    trimmed_s = completion_formatter(replaced_s)
                    formatted_completions.append(trimmed_s)
                writer.writerow([i, rel, head, tail, formatted_completions])

    # 参照文あり
    else:
        with open(output_path_csv, 'w') as f:
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
    print(f"Successfully dumped {output_path_csv}")


# result_formmaterによって出力されたテキストファイルに対し，手動でラベル付したものを読み込んでリスト形式に変換
def convert_formatted_results_tolist(input_path: str, num_return_sequences=30, num_pairs=25):
    """output_data
    0) rel: str
    1) head: str
    2) tail: str
    3) completions: List[str]
    4) labels: List[int]
    """
    with open(input_path, 'r') as f:
        input_data = [line for line in f if line.strip()]
        assert len(input_data) == (num_return_sequences + 1) * num_pairs

    output_data = []
    for i in range(num_pairs):
        rel, head, tail = (input_data[(num_return_sequences+1)*i]).split(", ")
        tail = tail.rstrip("\n")
        labels = [int(input_data[j][0]) for j in range((num_return_sequences+1)*i+1, (num_return_sequences+1)*(i+1))]
        assert len(labels) == num_return_sequences
        completions = [(input_data[j][2:]).rstrip("\n") for j in range((num_return_sequences+1)*i+1, (num_return_sequences+1)*(i+1))]
        output_data.append([rel, head, tail, completions, labels])
    return output_data


if __name__ == "__main__":
    result_dir = "results/ja/連想語頻度表/evaluation/master/231115101840_12"
    input_path = f"{result_dir}/generated_texts.csv"
    output_path_txt = f"{result_dir}/formatted_results.txt"
    output_path_csv = f"{result_dir}/formatted_results.csv"
    num_refs = 0
    template_path = "datasets/連想語頻度表/templates/5-shot_len_deduplicated.json"
    model = "rinna/japanese-gpt-neox-3.6b"
    result_formatter(input_path, output_path_txt, output_path_csv, num_refs, template_path)
