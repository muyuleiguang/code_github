# Verbatim Memorization under Instruction Fine-Tuning (OLMo-2)

This repository contains code for studying the impact of instruction fine-tuning (SFT) through the lens of **verbatim memorization** of the pre-training corpus.

## Why OLMo-2?
We require (i) an open-source base model, (ii) an official SFT variant from the same base model, (iii) public access to the corresponding pre-training data and SFT mixtures, and (iv) multiple parameter scales.  
**OLMo-2** satisfies all four criteria (base + official SFT + public corpora + multiple scales).

## Models (Base and SFT)
- `allenai/OLMo-2-0425-1B`
- `allenai/OLMo-2-1124-7B`
- `allenai/OLMo-2-1124-13B`
- `allenai/OLMo-2-0325-32B`

## Evaluation Data
Memorization test sets are constructed from the OLMo-2 second-stage pre-training corpus **Dolmino-Mix-1124**, focusing on:
- **StackExchange (Q&A-format)**, **DCLM-privacy (safety/privacy)**, and **Wikipedia (factual knowledge)**.  
We additionally include **Idioms** for context-independent memorization probing, and evaluate downstream benchmarks covering **Reasoning (GSM8K, MATH)** and **Knowledge (MMLU, PopQA)**.

## Repository Layout
- `exp_1/`: memorization generation + memorization/style analyses  
- `exp_2/`: downstream evaluation + memorization–downstream relationship analysis  
- `instruction_test_data/`: instruction-data pipeline  
- `pretraining_test_data/`: pre-training test-data pipeline  
- `plot/`: plotting + correlation metrics  

## Running the Code
Full reproducibility commands are provided in:
- `run.md` (main experiments)
- `instruction_test_data/run_instruction_data.md`
- `pretraining_test_data/run.md`

Minimal templates (run from repo root):

```bash
# 1) Memorization generation (Base vs SFT)
python exp_1/generation_2.py --datasets stackexchange dclm-privacy wiki-fact \
  --model_name <MODEL_NAME> --model_type <base|sft> --max_new_tokens 128

# 2) Memorization scoring (one dataset + one model scale per run)
python exp_1/analysis_3.py --prefix_lengths 16 --max_samples 100 \
  --datasets <DATASET> --model_scale <1B|7B|13B|32B>

# 3) Downstream evaluation
python exp_2/evaluate_downstream_tasks_1.py --num_samples 100 --few_shot_count 5 --batch_size <BATCH> --datasets <mmlu|popqa>

# 4) Memorization vs downstream relationship
python exp_2/analyze_memorization_downstream_relationship_2.py
# Verbatim Memorization under Instruction Fine-Tuning (OLMo-2)

This repository contains code for studying the impact of instruction fine-tuning (SFT) through the lens of **verbatim memorization** of the pre-training corpus.

## Why OLMo-2?
We require (i) an open-source base model, (ii) an official SFT variant from the same base model, (iii) public access to the corresponding pre-training data and SFT mixtures, and (iv) multiple parameter scales.  
**OLMo-2** satisfies all four criteria (base + official SFT + public corpora + multiple scales).

## Models (Base and SFT)
- `allenai/OLMo-2-0425-1B`
- `allenai/OLMo-2-1124-7B`
- `allenai/OLMo-2-1124-13B`
- `allenai/OLMo-2-0325-32B`

## Evaluation Data
Memorization test sets are constructed from the OLMo-2 second-stage pre-training corpus **Dolmino-Mix-1124**, focusing on:
- **StackExchange (Q&A-format)**, **DCLM-privacy (safety/privacy)**, and **Wikipedia (factual knowledge)**.  
We additionally include **Idioms** for context-independent memorization probing, and evaluate downstream benchmarks covering **Reasoning (GSM8K, MATH)** and **Knowledge (MMLU, PopQA)**.

## Repository Layout
- `exp_1/`: memorization generation + memorization/style analyses  
- `exp_2/`: downstream evaluation + memorization–downstream relationship analysis  
- `instruction_test_data/`: instruction-data pipeline  
- `pretraining_test_data/`: pre-training test-data pipeline  
- `plot/`: plotting + correlation metrics  

## Running the Code
Full reproducibility commands are provided in:
- `run.md` (main experiments)
- `instruction_test_data/run_instruction_data.md`
- `pretraining_test_data/run.md`

Minimal templates (run from repo root):

```bash
# 1) Memorization generation (Base vs SFT)
python exp_1/generation_2.py --datasets stackexchange dclm-privacy wiki-fact \
  --model_name <MODEL_NAME> --model_type <base|sft> --max_new_tokens 128

# 2) Memorization scoring (one dataset + one model scale per run)
python exp_1/analysis_3.py --prefix_lengths 16 --max_samples 100 \
  --datasets <DATASET> --model_scale <1B|7B|13B|32B>

# 3) Downstream evaluation
python exp_2/evaluate_downstream_tasks_1.py --num_samples 100 --few_shot_count 5 --batch_size <BATCH> --datasets <mmlu|popqa>

# 4) Memorization vs downstream relationship
python exp_2/analyze_memorization_downstream_relationship_2.py
Notes

This repo provides code and instructions. Please follow the original licenses/terms for models and datasets.

Citation

If you use this code, please cite our paper:
Instruction Fine-Tuning through the Lens of Verbatim Memorization
