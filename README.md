# Verbatim Memorization under Instruction Fine-Tuning (OLMo-2)

This repository contains code for studying the impact of instruction fine-tuning (SFT) through the lens of **verbatim memorization** of the pre-training corpus.

## Why OLMo-2?

Our goal requires models that provide: (i) an open-source base model, (ii) an official SFT variant from the same base model, (iii) public access to the corresponding pre-training data and SFT mixtures, and (iv) multiple parameter scales.  
**OLMo-2** satisfies all four criteria (base + official SFT + public corpora + multiple scales).

**Models used (Base and SFT):**

- `allenai/OLMo-2-0425-1B`
- `allenai/OLMo-2-1124-7B`
- `allenai/OLMo-2-1124-13B`
- `allenai/OLMo-2-0325-32B`

## Evaluation Data

### Memorization test sets (from Dolmino-Mix-1124)

We construct memorization test sets directly from the OLMo-2 second-stage pre-training corpus **Dolmino-Mix-1124**, focusing on three categories:

- **StackExchange (Q&A-format)**: instruction-like prompts filtered from high-quality posts and interrogative cues.
- **DCLM-privacy (safety/privacy)**: passages containing sensitive patterns (e.g., emails/phones/addresses).
- **Wikipedia (factual knowledge)**: encyclopedic text for knowledge memorization.

We sample documents without replacement and extract fixed-length segments for prefix–continuation memorization evaluation.

### Context-independent memorization

- **Idioms**: predict the final word given the first *n−1* words (exact-match accuracy), following prior idiom-based memorization probes.

### Downstream benchmarks

We evaluate both reasoning- and knowledge-intensive performance:

- **Reasoning**: GSM8K, MATH
- **Knowledge**: MMLU, PopQA

## Repository Layout

- `exp_1/`: generation + memorization/style analyses (and LaTeX table utilities)
- `exp_2/`: downstream evaluation + memorization–downstream relationship analysis
- `instruction_test_data/`: instruction-data pipeline (download/analyze/feature/match/build)
- `pretraining_test_data/`: pre-training test-data pipeline (download/build/analyze)
- `plot/`: plotting + correlation metrics
- `run.md`: main run commands used in experiments

## Running the Code

For exact commands, see:

- `run.md`
- `instruction_test_data/run_instruction_data.md`
- `pretraining_test_data/run.md`

Typical examples (from repo root):

```bash
# Memorization generation (Base vs SFT)
python exp_1/generation_2.py --datasets stackexchange dclm-privacy wiki-fact \
  --model_name allenai_OLMo-2-1124-7B --model_type base --max_new_tokens 128

python exp_1/generation_2.py --datasets stackexchange dclm-privacy wiki-fact \
  --model_name allenai_OLMo-2-1124-7B --model_type sft --max_new_tokens 128

# Downstream evaluation
python exp_2/evaluate_downstream_tasks_1.py --num_samples 100 --few_shot_count 5 --batch_size 16 --datasets mmlu
python exp_2/evaluate_downstream_tasks_1.py --num_samples 100 --few_shot_count 5 --batch_size 4  --datasets popqa

# Memorization vs downstream relationship
python exp_2/analyze_memorization_downstream_relationship_2.py
```

## Notes

This repo provides code and instructions. Please follow the original licenses/terms for models and datasets.

## Citation

If you use this code, please cite our paper:

*Instruction Fine-Tuning through the Lens of Verbatim Memorization*

