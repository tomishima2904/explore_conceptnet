SEED=1
RESULT_DIR=results/ja/連想語頻度表/text_generation/231031120400_dev30_M10_0S_0R
MODEL_NAME=Matsuo10
INTRA_SELECTION_OPTION=len
NUM_PAIRS=25
NUM_RETURN_SEQUENCES=30

# 自動評価
python '/work/tomishima2904/explore_conceptnet/src/models/execution_accuracy.py' \
    --seed=$SEED --result_dir=$RESULT_DIR
# 手動評価と自動評価の相関を調べる
python '/work/tomishima2904/explore_conceptnet/src/utils/m_and_a_comparator.py' \
    --seed=$SEED \
    --result_dir=$RESULT_DIR \
    --model_name=$MODEL_NAME \
    --intra_selection_option=$INTRA_SELECTION_OPTION \
    --num_pairs=$NUM_PAIRS \
    --num_return_sequences=$NUM_RETURN_SEQUENCES
