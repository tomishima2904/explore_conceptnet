SEED=1
# テスト用の連想理由の説明（参照文無し）
python '/work/tomishima2904/explore_conceptnet/src/models/text_generation_model.py'  \
        --device_type=cuda:0 \
        --model=matsuo-lab/weblab-10b \
        --num_return_sequences=3 \
        --num_refs=0 \
        --template_dir=datasets/連想語頻度表/templates \
        --template_name="10-shot_len_xRandom" \
        --seed=$SEED

# テスト用の連想理由の説明（参照文有り for ICNLSP2023）
# python '/work/tomishima2904/explore_conceptnet/src/models/text_generation_model.py'  \
#         --device_type=cuda:0 \
#         --model=rinna/japanese-gpt-neox-3.6b \
#         --num_refs=0 \
#         --template_name=zero-shot_no_refs_5 \
#         --seed=$SEED

# # Few-shot用（訓練用）の連想理由の説明
# python '/work/tomishima2904/explore_conceptnet/src/models/text_generation_model.py' \
#         --device_type=cuda:0 \
#         --model=matsuo-lab/weblab-10b \
#         --num_return_sequences=30 \
#         --num_refs=0 \
#         --template_name=zero-shot_no_refs_5 \
#         --seed=$SEED \
#         --input_path="datasets/連想語頻度表/all/dev_mini2/htrkpns_tmp.csv.gz"

# rinna/japanese-gpt-neox-3.6b
# cyberagent/open-calm-7b
# cyberagent/calm2-7b
# matsuo-lab/weblab-10b
# pfnet/plamo-13b
