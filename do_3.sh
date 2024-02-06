SEED=1
# テスト用の連想理由の説明（参照文無し）
python '/work/tomishima2904/explore_conceptnet/src/models/text_generation_model.py'  \
        --device_type=cuda:0 \
        --dir_name="evaluation/master_alpha" \
        --model=matsuo-lab/weblab-10b \
        --num_pre_return_sequences=30 \
        --num_return_sequences=3 \
        --num_refs=0 \
        --template_dir=datasets/連想語頻度表/templates \
        --template_name="10-shot_len_xRandom" \
        --seed=$SEED
