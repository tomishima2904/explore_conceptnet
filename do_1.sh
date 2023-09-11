SEED=8
# python '/work/tomishima2904/explore_conceptnet/src/models/text_generation_model.py'  --device_type=cuda:1 --num_refs=0 --template_name=zero-shot_no_refs_3 --seed=$SEED
python '/work/tomishima2904/explore_conceptnet/src/models/text_generation_model.py'  --device_type=cuda:1 --num_refs=3 --template_name=zero-shot_with_refs_3 --seed=$SEED
# python '/work/tomishima2904/explore_conceptnet/src/models/text_generation_model.py'  --device_type=cuda:1 --num_refs=30 --template_name=zero-shot_with_refs_3 --seed=$SEED
SEED=10
python '/work/tomishima2904/explore_conceptnet/src/models/text_generation_model.py'  --device_type=cuda:1 --num_refs=3 --template_name=zero-shot_with_refs_3 --seed=$SEED
