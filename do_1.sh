# テスト用の連想理由の説明（参照文無し）
python '/work/tomishima2904/explore_conceptnet/src/models/text_generation_model.py'  \
        --device_type=cuda:1 \
        --model=matsuo-lab/weblab-10b \
        --num_return_sequences=3 \
        --num_refs=0 \
        --template_name=few-shot_no_refs_5 \
        --seed=$SEED

# rinna/japanese-gpt-neox-3.6b
# cyberagent/open-calm-7b
# matsuo-lab/weblab-10b
# pfnet/plamo-13b
