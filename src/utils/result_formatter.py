import csv
import re

def result_formatter(result_dir: str, num_refs: int):
    input_path = f"{result_dir}/generated_texts.csv"

    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        all_data = [row for row in reader]

    # 0-shotかつ参照文なし
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
                    f.write(f"{s}\n")
                f.write("\n")
    #  0-shotかつ参照文あり
    else:
        with open(output_path, 'w') as f:
            for row in all_data:
                head = row[0]
                tail = row[1]
                rel = row[2]
                generated_sentences = eval(row[-1])

                f.write(f"{rel}, {head}, {tail}\n")

                for i, generated_sentence in enumerate(generated_sentences):
                    reference_position = generated_sentence.index(reference_text)
                    after_text = generated_sentence[reference_position + len(reference_text):]
                    f.write(after_text)
                    f.write("\n")
                f.write("\n")
    print(f"Successfully dumped {output_path}")

    output_path = f"{result_dir}/formatted_results.csv"

    # 0-shotかつ参照文なし (csv出力)
    if num_refs == 0:
        with open(output_path, 'w') as f:
            writer = csv.writer(f)
            for i, row in enumerate(all_data):
                head = row[0]
                tail = row[1]
                rel = eval(row[2])[0]
                generated_sentences = eval(row[-1])
                writer.writerow([i, rel, head, tail, generated_sentences])

    # 0-shotかつ参照文あり (csv出力)
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
                    reference_position = generated_sentence.index(reference_text)
                    after_text = generated_sentence[reference_position + len(reference_text):]
                    texts.append(after_text)
                writer.writerow([j, rel, head, tail, texts])
    print(f"Successfully dumped {output_path}")
