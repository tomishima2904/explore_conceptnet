SEED=1

# 連想理由生成を行った後，提案手法によるスコアリングをして選定を行う

python '/work/tomishima2904/explore_conceptnet/src/models/gen_with_mlm_selection.py'  \
        --device_type=cuda:0 \
        --dir_name="evaluation/master_alpha" \
        --model=matsuo-lab/weblab-10b \
        --num_pre_return_sequences=30 \
        --num_return_sequences=3 \
        --num_refs=0 \
        --template_dir=datasets/連想語頻度表/templates/master30 \
        --template_name="5-shot_len_dedupulicated" \
        --seed=$SEED

# template_name
# - 5-shot_len_deduplicated: 提案手法
# - 5-shot_len_deduplicated_random: 自動ランダム
# - 5-shot_len_mR: 手動1 (富島)
# - 5-shot_len_mT: 手動2 (問井)
# - 5-shot_len_mY: 手動3 (悠真)

# 出力結果は `results/ja/連想語頻度表/{dir_name}` に保存
# 実験に使うtemplateは `datasets/連想語頻度表/templates/master30` にある
# 上記の `{template_name}.json` が `template_dir` に保存されている
