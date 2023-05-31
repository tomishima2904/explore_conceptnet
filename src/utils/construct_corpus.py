import gzip
import csv

import file_handlers as fh
from extract_entity import extract_entity


# ConceptNetのデータセットに head entity と tail entity を含む文を追加する
def construct_corpus(
        output_path: str,
        conceptnet: list,
        wiki_corpus: dict,
        stair_caps_corpus: dict):
    """
    conceptnet: n×3のリスト (relation, head, tail)

    下記コーパスは`head`と`tail`を含む文が最大3文ある
    wiki_corpus: wiki_corpus[head][tail][:3]
    stair_caps_corpus: wiki_corpus[head][tail][:3]
    """
    with gzip.open(output_path, "wt") as wf:
        writer = csv.writer(wf)
        for row in conceptnet:
            relation = row[0]
            head_uri = row[1]
            tail_uri = row[2]
            head = extract_entity(lang="ja", uri=head_uri)
            tail = extract_entity(lang="ja", uri=tail_uri)

            if head == None or tail == None:
                continue

            sentences = []
            sentences.extend(wiki_corpus[head][tail])
            sentences.extend(stair_caps_corpus[head][tail])

        writer.writerow((relation, head_uri, tail_uri, sentences))
    print(f"Successfully dumped {output_path} !")


# STAIR Captions から caption を抽出するためだけに使用
def _stair_captions_to_csv(input_path: str, output_path: str) -> None:
    json_data = fh.read_json(input_path)
    stair_captions = json_data["annotations"]
    with open(output_path, "w") as f:
        for data in stair_captions:
            print(data["caption"], file=f)
    print(f"Successfully dumped {output_path} !")


if __name__ == "__main__":
    wiki_corpus_path = "datasets/jawiki-20221226/extracted_corpus.json"
    wiki_corpus = fh.read_json(wiki_corpus_path)

    stair_caps_corpus_path = "datasets/STAIR-captions/extracted_corpus.json"
    stair_caps_corpus = fh.read_json(stair_caps_corpus_path)

    # ConceptNetをロード
    lang = "ja"
    dataset_dir = f"datasets/conceptnet-assertions-5.7.0/{lang}"
    input_file = f"conceptnet-assertions-5.7.0_{lang}.csv.gz"
    conceptnet_path = f"{dataset_dir}/{input_file}"

    conceptnet = []
    with gzip.open(conceptnet_path, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            conceptnet.append(row[1:-1])  # relation と head と tail だけ取得

    output_path = f"{dataset_dir}/conceptnet-assertions-5.7.0_{lang}_with_sentences.csv.gz"

    construct_corpus(output_path, conceptnet, wiki_corpus, stair_caps_corpus)


    # # STAIR Captions データセットからキャプションだけを抽出
    # stair_captions_dir = f"{dataset_dir}/STAIR-captions"
    # input_path = f"{stair_captions_dir}/stair_captions_v1.2_train.json"
    # output_path = f"{stair_captions_dir}/stair_captions_v1.2_train_sentences.txt"
    # _stair_captions_to_csv(input_path, output_path)


