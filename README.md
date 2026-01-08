# Verbatim Memorization under Instruction Fine-Tuning (OLMo-2)

This repository contains code for studying the impact of instruction fine-tuning (SFT) through the lens of **verbatim memorization** of the pre-training corpus.

## Why OLMo-2?
Our goal requires models that provide: (i) an open-source base model, (ii) an official SFT variant from the same base model, (iii) public access to the corresponding pre-training data and SFT mixtures, and (iv) multiple parameter scales.  
**OLMo-2** satisfies all four criteria (base + official SFT + public corpora + multiple scales).

## Models (Base and SFT)
- `allenai/OLMo-2-0425-1B`
- `allenai/OLMo-2-1124-7B`
- `allenai/OLMo-2-1124-13B`
- `allenai/OLMo-2-0325-32B`

## Evaluation Data

### Memorization test sets (from Dolmino-Mix-1124)
We construct memorization test sets directly from the OLMo-2 second-stage pre-training corpus **Dolmino-Mix-1124**, focusing on:
- **StackExchange (Q&A-format)**: instruction-like prompts filtered from high-quality posts and interrogative cues.
- **DCLM-privacy (safety/privacy)**: passages containing sensitive patterns (e.g., emails/phones/addresses).
- **Wikipedia (factual knowledge)**: encyclopedic text for knowledge memorization.

We sample documents without replacement and extract fixed-length segments for prefix–continuation memorization evaluation.

### Context-independent memorization
- **Idioms**: predict the final word given the first *n−1* words (exact-match accuracy), following prior idiom-based memorization probes.

### Downstream benchmarks
- **Reasoning**: GSM8K, MATH  
- **Knowledge**: MMLU, PopQA

## Repository Layout
- `exp_1/`: memorization generation + memorization/style analyses
- `exp_2/`: downstream evaluation + memorization–downstream relationship analysis
- `instruction_test_data/`: instruction-data pipeline (download/analyze/feature/match/build)
- `pretraining_test_data/`: pre-training test-data pipeline (download/build/analyze)
- `plot/`: plotting + correlation metrics
- `run.md`: full run commands used in experiments

## Running the Code

### 1) Memorization test data generation (Base vs SFT)
Run generation for each model scale:

```bash
# 1B
python exp_1/generation_2.py --datasets stackexchange dclm-privacy wiki-fact \
  --model_name allenai_OLMo-2-0425-1B --model_type base --max_new_tokens 128
python exp_1/generation_2.py --datasets stackexchange dclm-privacy wiki-fact \
  --model_name allenai_OLMo-2-0425-1B --model_type sft  --max_new_tokens 128

# 7B
python exp_1/generation_2.py --datasets stackexchange dclm-privacy wiki-fact \
  --model_name allenai_OLMo-2-1124-7B --model_type base --max_new_tokens 128
python exp_1/generation_2.py --datasets stackexchange dclm-privacy wiki-fact \
  --model_name allenai_OLMo-2-1124-7B --model_type sft  --max_new_tokens 128

# 13B
python exp_1/generation_2.py --datasets stackexchange dclm-privacy wiki-fact \
  --model_name allenai_OLMo-2-1124-13B --model_type base --max_new_tokens 128
python exp_1/generation_2.py --datasets stackexchange dclm-privacy wiki-fact \
  --model_name allenai_OLMo-2-1124-13B --model_type sft  --max_new_tokens 128

# 32B
python exp_1/generation_2.py --datasets stackexchange dclm-privacy wiki-fact \
  --model_name allenai_OLMo-2-0325-32B --model_type base --max_new_tokens 128
python exp_1/generation_2.py --datasets stackexchange dclm-privacy wiki-fact \
  --model_name allenai_OLMo-2-0325-32B --model_type sft  --max_new_tokens 128
