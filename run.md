# Memorization Test Data Generation

## 1B

python generation_2.py  --datasets stackexchange dclm-privacy wiki-fact --model_name allenai_OLMo-2-0425-1B --model_type base  --max_new_tokens 128  # base model
python generation_2.py  --datasets stackexchange dclm-privacy wiki-fact --model_name allenai_OLMo-2-0425-1B --model_type sft   --max_new_tokens 128  # SFT model

## 7B

python generation_2.py  --datasets stackexchange dclm-privacy wiki-fact --model_name allenai_OLMo-2-1124-7B  --model_type base  --max_new_tokens 128  # base model
python generation_2.py  --datasets stackexchange dclm-privacy wiki-fact --model_name allenai_OLMo-2-1124-7B  --model_type sft   --max_new_tokens 128  # SFT model

## 13B

python generation_2.py  --datasets stackexchange dclm-privacy wiki-fact --model_name allenai_OLMo-2-1124-13B --model_type base  --max_new_tokens 128  # base model
python generation_2.py  --datasets stackexchange dclm-privacy wiki-fact --model_name allenai_OLMo-2-1124-13B --model_type sft   --max_new_tokens 128  # SFT model

## 32B

python generation_2.py  --datasets stackexchange dclm-privacy wiki-fact --model_name allenai_OLMo-2-0325-32B --model_type base  --max_new_tokens 128  # base model
python generation_2.py  --datasets stackexchange dclm-privacy wiki-fact --model_name allenai_OLMo-2-0325-32B --model_type sft   --max_new_tokens 128  # SFT model


# Analyze the generation results of the memorization test to obtain memorization scores.
# Note: Each run can process only one dataset and one model.

python analysis_3.py   --prefix_lengths 16  --max_samples 100  --datasets stackexchange --model_scale 1B
python analysis_3.py   --prefix_lengths 16  --max_samples 100  --datasets dclm-privacy  --model_scale 7B
python analysis_3.py   --prefix_lengths 16  --max_samples 100  --datasets wiki-fact     --model_scale 13B
python analysis_3.py   --prefix_lengths 16  --max_samples 100  --datasets wiki-fact     --model_scale 32B


# Generate downstream task results

python evaluate_downstream_tasks_1.py  --num_samples 100 --few_shot_count 5  --batch_size 16  --datasets mmlu
python evaluate_downstream_tasks_1.py  --num_samples 100 --few_shot_count 5  --batch_size 4   --datasets popqa


# Analyze the relationship between memorization and downstream performance

python analyze_memorization_downstream_relationship_2.py