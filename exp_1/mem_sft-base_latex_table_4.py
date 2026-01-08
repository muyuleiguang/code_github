#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import os
import glob
import pandas as pd
from typing import Dict, List
import warnings
import numpy as np
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import editdistance

warnings.filterwarnings('ignore')


class MemorizationMetrics:
    def __init__(
            self,
            tokenizer_name: str = None,
            sentence_model_name: str = "/root/autodl-tmp/ift_memorization/model_cache/sentence_transformers"
    ):
        """
        Initialize evaluation metrics.

        Args:
            tokenizer_name: Tokenizer model name (optional).
            sentence_model_name: Sentence embedding model name/path.
        """
        self.tokenizer = None
        if tokenizer_name:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except:
                print(f"警告: 无法加载tokenizer {tokenizer_name}")

        try:
            self.sentence_model = SentenceTransformer(sentence_model_name)
        except Exception as e:
            print(f"警告: 加载 sentence model '{sentence_model_name}' 时发生错误。")
            self.sentence_model = None

        self.rouge = Rouge()
        self.smoothing = SmoothingFunction()

    def exact_match_rate(
            self,
            generated_tokens: List[List[int]],
            reference_tokens: List[List[int]]
    ) -> Dict[str, float]:
        """
        Method 1: Exact Match Rate (EMR)

        The proportion of samples where the generated content exactly matches the training
        reference (verbatim). This is the strictest metric.

        Args:
            generated_tokens: List of generated token sequences.
            reference_tokens: List of reference token sequences.

        Returns:
            Exact match metrics.
        """
        assert len(generated_tokens) == len(reference_tokens)

        exact_matches = 0
        for gen_tokens, ref_tokens in zip(generated_tokens, reference_tokens):
            if gen_tokens == ref_tokens:
                exact_matches += 1

        return {
            "exact_match_rate": exact_matches / len(generated_tokens),
            "exact_matches": exact_matches,
            "total_samples": len(generated_tokens)
        }

    def rouge_bleu_scores(
            self,
            generated_texts: List[str],
            reference_texts: List[str],
            generated_tokens: List[List[int]] = None,
            reference_tokens: List[List[int]] = None
    ) -> Dict[str, float]:
        """
        Method 2: ROUGE / BLEU Scores

        Measures n-gram overlap between generated text and reference text. This can capture
        near-verbatim memorization.

        Args:
            generated_texts: List of generated texts.
            reference_texts: List of reference texts.
            generated_tokens: List of generated token sequences (optional, for token-level BLEU).
            reference_tokens: List of reference token sequences (optional, for token-level BLEU).

        Returns:
            ROUGE and BLEU metrics.
        """
        results = {}

        # ROUGE scores (text-based)
        if generated_texts and reference_texts:
            assert len(generated_texts) == len(reference_texts)

            rouge_1_scores = []
            rouge_2_scores = []
            rouge_l_scores = []

            for gen, ref in zip(generated_texts, reference_texts):
                if not gen or not ref:
                    rouge_1_scores.append(0.0)
                    rouge_2_scores.append(0.0)
                    rouge_l_scores.append(0.0)
                    continue

                try:
                    scores = self.rouge.get_scores(gen, ref)[0]
                    rouge_1_scores.append(scores['rouge-1']['f'])
                    rouge_2_scores.append(scores['rouge-2']['f'])
                    rouge_l_scores.append(scores['rouge-l']['f'])
                except:
                    rouge_1_scores.append(0.0)
                    rouge_2_scores.append(0.0)
                    rouge_l_scores.append(0.0)

            results.update({
                "rouge_1_f": np.mean(rouge_1_scores),
                "rouge_2_f": np.mean(rouge_2_scores),
                "rouge_l_f": np.mean(rouge_l_scores),
                "rouge_1_std": np.std(rouge_1_scores),
                "rouge_2_std": np.std(rouge_2_scores),
                "rouge_l_std": np.std(rouge_l_scores)
            })

        # BLEU scores (token-based, more precise)
        if generated_tokens and reference_tokens:
            assert len(generated_tokens) == len(reference_tokens)

            bleu_1_scores = []
            bleu_2_scores = []
            bleu_4_scores = []

            for gen_tokens, ref_tokens in zip(generated_tokens, reference_tokens):
                if not gen_tokens or not ref_tokens:
                    bleu_1_scores.append(0.0)
                    bleu_2_scores.append(0.0)
                    bleu_4_scores.append(0.0)
                    continue

                # Convert token IDs to strings for BLEU computation
                gen_str_tokens = [str(t) for t in gen_tokens]
                ref_str_tokens = [str(t) for t in ref_tokens]

                # BLEU-1
                bleu_1 = sentence_bleu([ref_str_tokens], gen_str_tokens,
                                       weights=(1, 0, 0, 0),
                                       smoothing_function=self.smoothing.method1)
                bleu_1_scores.append(bleu_1)

                # BLEU-2
                bleu_2 = sentence_bleu([ref_str_tokens], gen_str_tokens,
                                       weights=(0.5, 0.5, 0, 0),
                                       smoothing_function=self.smoothing.method1)
                bleu_2_scores.append(bleu_2)

                # BLEU-4
                bleu_4 = sentence_bleu([ref_str_tokens], gen_str_tokens,
                                       weights=(0.25, 0.25, 0.25, 0.25),
                                       smoothing_function=self.smoothing.method1)
                bleu_4_scores.append(bleu_4)

            results.update({
                "bleu_1": np.mean(bleu_1_scores),
                "bleu_2": np.mean(bleu_2_scores),
                "bleu_4": np.mean(bleu_4_scores),
                "bleu_1_std": np.std(bleu_1_scores),
                "bleu_2_std": np.std(bleu_2_scores),
                "bleu_4_std": np.std(bleu_4_scores)
            })

        return results

    def edit_distance_metrics(
            self,
            generated_tokens: List[List[int]],
            reference_tokens: List[List[int]],
            generated_texts: List[str] = None,
            reference_texts: List[str] = None
    ) -> Dict[str, float]:
        """
        Method 3: Edit Distance

        The number of insertions/deletions/substitutions needed to transform the generated
        output into the reference. Smaller distance indicates stronger memorization.
        This primarily uses token-level edit distance.

        Args:
            generated_tokens: List of generated token sequences.
            reference_tokens: List of reference token sequences.
            generated_texts: List of generated texts (optional).
            reference_texts: List of reference texts (optional).

        Returns:
            Edit distance metrics.
        """
        assert len(generated_tokens) == len(reference_tokens)

        token_distances = []
        normalized_token_distances = []

        # Token-level edit distance (primary metric)
        for gen_tokens, ref_tokens in zip(generated_tokens, reference_tokens):
            token_dist = editdistance.eval(gen_tokens, ref_tokens)
            token_distances.append(token_dist)

            # Normalized token edit distance
            max_token_len = max(len(gen_tokens), len(ref_tokens))
            if max_token_len > 0:
                normalized_token_distances.append(token_dist / max_token_len)
            else:
                normalized_token_distances.append(0.0)

        results = {
            "token_edit_distance": np.mean(token_distances),
            "token_edit_distance_std": np.std(token_distances),
            "normalized_token_edit_distance": np.mean(normalized_token_distances),
            "normalized_token_edit_distance_std": np.std(normalized_token_distances),
            "min_token_edit_distance": np.min(token_distances),
            "max_token_edit_distance": np.max(token_distances),
            "median_token_edit_distance": np.median(token_distances)
        }

        # Character-level edit distance (if texts are provided)
        if generated_texts and reference_texts:
            char_distances = []
            normalized_char_distances = []

            for gen_text, ref_text in zip(generated_texts, reference_texts):
                char_dist = editdistance.eval(gen_text, ref_text)
                char_distances.append(char_dist)

                # Normalized character edit distance
                max_char_len = max(len(gen_text), len(ref_text))
                if max_char_len > 0:
                    normalized_char_distances.append(char_dist / max_char_len)
                else:
                    normalized_char_distances.append(0.0)

            results.update({
                "char_edit_distance": np.mean(char_distances),
                "char_edit_distance_std": np.std(char_distances),
                "normalized_char_edit_distance": np.mean(normalized_char_distances),
                "normalized_char_edit_distance_std": np.std(normalized_char_distances),
                "min_char_edit_distance": np.min(char_distances),
                "max_char_edit_distance": np.max(char_distances),
                "median_char_edit_distance": np.median(char_distances)
            })

        return results

    def semantic_similarity(
            self,
            generated_texts: List[str],
            reference_texts: List[str]
    ) -> Dict[str, float]:
        """
        Method 4: Semantic Similarity

        Compute embedding-based similarity (e.g., via Sentence-BERT).

        Args:
            generated_texts: List of generated texts.
            reference_texts: List of reference texts.

        Returns:
            Semantic similarity metrics.
        """
        if self.sentence_model is None:
            return {
                "avg_similarity": 0.0,
                "similarity_std": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "median_similarity": 0.0
            }

    def likelihood_ppl_loss_metrics(
            self,
            top_tokens_list: List[List[List[Dict]]],
            reference_tokens: List[List[int]],
            logits=None
    ) -> Dict[str, float]:
        """
        Method 5: Likelihood, PPL, loss, logits

        Compute memorization-related metrics based on model output probabilities.

        Args:
            top_tokens_list: Top-k token probability information at each step for each sample
                Format: [sample][step][top_k_tokens]
                Each token_info contains: {'token_id': int, 'probability': float, 'rank': int}
            reference_tokens: List of reference token sequences.
            logits: Full logits tensor (optional; more precise if available).

        Returns:
            Likelihood / perplexity / loss related metrics.
        """
        if not top_tokens_list or not reference_tokens:
            return {
                "avg_log_likelihood": float('-inf'),
                "perplexity": float('inf'),
                "avg_loss": float('inf'),
                "target_token_probability": 0.0,
                "target_token_rank": float('inf'),
                "target_in_top1_rate": 0.0,
                "target_in_top3_rate": 0.0,
                "target_in_top5_rate": 0.0
            }

        log_likelihoods = []
        losses = []
        target_probs = []
        target_ranks = []
        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        total_positions = 0

        for sample_idx, (sample_top_tokens, ref_tokens) in enumerate(zip(top_tokens_list, reference_tokens)):
            if not sample_top_tokens or not ref_tokens:
                continue

            sample_log_likelihood = 0.0
            sample_positions = 0

            for step_idx, step_top_tokens in enumerate(sample_top_tokens):
                if step_idx >= len(ref_tokens):
                    break

                target_token = ref_tokens[step_idx]
                total_positions += 1
                sample_positions += 1

                # Find the target token's rank and probability in top-k
                target_found = False

                for rank, token_info in enumerate(step_top_tokens):
                    token_id = token_info.get('token_id')
                    prob = token_info.get('probability', 0.0)

                    if token_id == target_token:
                        target_probs.append(prob)
                        target_ranks.append(rank + 1)  # rank starts from 1
                        target_found = True

                        # Compute log likelihood
                        if prob > 0:
                            log_prob = np.log(prob)
                            sample_log_likelihood += log_prob

                            # Compute cross-entropy loss
                            loss = -log_prob
                            losses.append(loss)
                        else:
                            sample_log_likelihood += -100  # avoid log(0)
                            losses.append(100)

                        # Top-k hit rates
                        if rank == 0:  # top-1
                            top1_hits += 1
                        if rank < 3:  # top-3
                            top3_hits += 1
                        if rank < 5:  # top-5
                            top5_hits += 1
                        break

                if not target_found:
                    # If target token is not in top-k, use a very small probability
                    target_probs.append(1e-10)
                    target_ranks.append(float('inf'))
                    sample_log_likelihood += -100
                    losses.append(100)

            if sample_positions > 0:
                log_likelihoods.append(sample_log_likelihood / sample_positions)

        # Aggregate metrics
        avg_log_likelihood = np.mean(log_likelihoods) if log_likelihoods else float('-inf')
        perplexity = np.exp(-avg_log_likelihood) if avg_log_likelihood != float('-inf') else float('inf')
        avg_loss = np.mean(losses) if losses else float('inf')
        avg_target_prob = np.mean(target_probs) if target_probs else 0.0

        finite_ranks = [r for r in target_ranks if r != float('inf')]
        avg_target_rank = np.mean(finite_ranks) if finite_ranks else float('inf')

        return {
            "avg_log_likelihood": float(avg_log_likelihood),
            "perplexity": float(perplexity),
            "avg_loss": float(avg_loss),
            "target_token_probability": float(avg_target_prob),
            "target_token_rank": float(avg_target_rank),
            "target_in_top1_rate": top1_hits / total_positions if total_positions > 0 else 0.0,
            "target_in_top3_rate": top3_hits / total_positions if total_positions > 0 else 0.0,
            "target_in_top5_rate": top5_hits / total_positions if total_positions > 0 else 0.0,
            "target_prob_std": float(np.std(target_probs)) if target_probs else 0.0,
            "total_positions": total_positions
        }

    def compute_all_metrics_from_data(
            self,
            samples: List[Dict]
    ) -> Dict[str, Dict]:
        """
        Compute all 5 metric families from sample data.

        Args:
            samples: Sample list; each sample contains fields such as generated_tokens,
                     original_continuation_tokens, etc.

        Returns:
            A dictionary containing results for all metrics.
        """
        # Extract data
        generated_tokens = []
        reference_tokens = []
        generated_texts = []
        reference_texts = []
        top_tokens_list = []

        for sample in samples:
            if 'generated_tokens' in sample and 'original_continuation_tokens' in sample:
                generated_tokens.append(sample['generated_tokens'])
                reference_tokens.append(sample['original_continuation_tokens'])

            if 'generated_text' in sample and 'original_continuation' in sample:
                generated_texts.append(sample['generated_text'])
                reference_texts.append(sample['original_continuation'])

            if 'top_tokens' in sample:
                top_tokens_list.append(sample['top_tokens'])

        results = {}

        # Method 1: Exact Match Rate
        if generated_tokens and reference_tokens:
            results["exact_match"] = self.exact_match_rate(generated_tokens, reference_tokens)

        # Method 2: ROUGE/BLEU scores
        if generated_texts and reference_texts:
            results["rouge_bleu"] = self.rouge_bleu_scores(
                generated_texts, reference_texts, generated_tokens, reference_tokens
            )

        # Method 3: Edit distance
        if generated_tokens and reference_tokens:
            results["edit_distance"] = self.edit_distance_metrics(
                generated_tokens, reference_tokens, generated_texts, reference_texts
            )

        # Method 4: Semantic similarity
        if generated_texts and reference_texts:
            results["semantic"] = self.semantic_similarity(generated_texts, reference_texts)

        # Method 5: Likelihood / PPL / loss
        if top_tokens_list and reference_tokens:
            results["likelihood"] = self.likelihood_ppl_loss_metrics(top_tokens_list, reference_tokens)

        return results


def sort_model_scales(model_scales):
    """
    Sort model scales by numeric size, e.g., 1B < 7B < 13B < 32B.

    Args:
        model_scales: List of model scale strings.

    Returns:
        sorted_scales: Sorted list of model scale strings.
    """

    def extract_scale_value(scale_str):
        """Extract the numeric value from a model scale string for sorting."""
        try:
            # Remove unit suffix (B, M, etc.)
            if scale_str.endswith('B'):
                return float(scale_str[:-1])
            elif scale_str.endswith('M'):
                return float(scale_str[:-1]) / 1000  # convert to B units
            else:
                # If no unit is present, treat as a raw number
                return float(scale_str)
        except:
            # If parsing fails, return a large value to place it at the end
            return float('inf')

    return sorted(model_scales, key=extract_scale_value)


def load_generation_results_memory_optimized(results_base_dir: str,
                                             model_scales: List[str],
                                             datasets: List[str] = None,
                                             prefix_lengths: List[int] = None,
                                             generation_lengths: List[int] = None,
                                             max_samples: int = None) -> Dict[str, Dict]:
    """
    Memory-optimized version: load generation result files in batches to reduce memory usage.

    Args:
        results_base_dir: Base directory containing exp1_generation_X subfolders.
        model_scales: List of model scales (e.g., ["1B", "7B", "13B"]).
        datasets: Datasets to load; None loads all.
        prefix_lengths: Prefix lengths to load; None loads all.
        generation_lengths: Generation lengths to load; None loads all.
        max_samples: Max samples per condition; default 50 to reduce memory usage.

    Returns:
        results_dict: Results organized by dataset, model scale, model type, prefix length, and generation length.
    """
    results_dict = {}

    # Default parameters - memory optimized
    if datasets is None:
        datasets = ['stackexchange', 'dclm-privacy', 'wiki-fact']
    if prefix_lengths is None:
        prefix_lengths = [16, 32, 64]
    if generation_lengths is None:
        generation_lengths = [8, 16, 128]
    if max_samples is None:
        max_samples = 50  # Memory optimization: default sample cap

    print(f"内存优化模式: 每个配置最多加载 {max_samples} 个样本")
    print(f"开始从基础目录加载数据: {results_base_dir}")

    # Iterate over folders for each generation_length
    for gen_length in generation_lengths:
        gen_folder = f"exp1_generation_{gen_length}"
        gen_dir = os.path.join(results_base_dir, gen_folder)

        if not os.path.exists(gen_dir):
            print(f"警告: 文件夹 {gen_dir} 不存在，跳过generation_length={gen_length}")
            continue

        print(f"\n处理generation_length={gen_length}的文件夹: {gen_folder}")

        # Find all jsonl files in this folder
        pattern = os.path.join(gen_dir, "*.jsonl")
        result_files = glob.glob(pattern)

        if not result_files:
            print(f"警告: 在 {gen_dir} 中未找到结果文件")
            continue

        print(f"在 {gen_folder} 中找到 {len(result_files)} 个结果文件")

        for filepath in result_files:
            try:
                # Parse info from filename
                filename = os.path.basename(filepath)
                print(f"正在处理文件: {filename}")

                filename_parts = filename.replace('.jsonl', '').split('_')

                if len(filename_parts) >= 5:
                    dataset = filename_parts[0]
                    prefix_info = filename_parts[1]
                    continuation_info = filename_parts[2]
                    file_model_scale = filename_parts[3]
                    model_type = filename_parts[4]

                    # Apply filters
                    if datasets and dataset not in datasets:
                        continue
                    if model_scales and file_model_scale not in model_scales:
                        continue

                    try:
                        prefix_length = int(prefix_info.replace('prefix', ''))
                        gen_length = int(continuation_info.replace('new', ''))
                    except:
                        continue

                    if prefix_lengths and prefix_length not in prefix_lengths:
                        continue

                    # Memory optimization: read file in a capped manner
                    samples = []
                    sample_count = 0

                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line_idx, line in enumerate(f):
                            if line.strip() and sample_count < max_samples:
                                try:
                                    sample = json.loads(line)
                                    samples.append(sample)
                                    sample_count += 1

                                except json.JSONDecodeError:
                                    continue

                    if not samples:
                        continue

                    # Organize data structure
                    if dataset not in results_dict:
                        results_dict[dataset] = {}
                    if file_model_scale not in results_dict[dataset]:
                        results_dict[dataset][file_model_scale] = {'base': {}, 'sft': {}}
                    if prefix_length not in results_dict[dataset][file_model_scale][model_type]:
                        results_dict[dataset][file_model_scale][model_type][prefix_length] = {}
                    if gen_length not in results_dict[dataset][file_model_scale][model_type][prefix_length]:
                        results_dict[dataset][file_model_scale][model_type][prefix_length][gen_length] = []

                    results_dict[dataset][file_model_scale][model_type][prefix_length][gen_length].extend(samples)
                    print(
                        f"✓ 加载 {dataset}-{file_model_scale}-{model_type}-prefix{prefix_length}-gen{gen_length}: {len(samples)} 条样本")

            except Exception as e:
                print(f"加载文件 {filepath} 时出错: {e}")
                continue

    print(f"\n内存优化加载完成!")
    return results_dict
    """
    Load generation result files for multiple datasets and model scales.
    Supports a folder structure where each generation_length corresponds to a subfolder.

    Args:
        results_base_dir: Base results directory containing exp1_generation_X subfolders.
        model_scales: List of model scales (e.g., ["1B", "7B", "13B", "32B"]).
        datasets: Datasets to load; None loads all.
        prefix_lengths: Prefix lengths to load; None loads all.
        generation_lengths: Generation lengths to load (continuation L); None loads all.
        max_samples: Max samples per condition; None loads all.

    Returns:
        results_dict: Results organized by dataset, model scale, model type, prefix length, and generation length.
        Format: {dataset: {model_scale: {model_type: {prefix_length: {generation_length: [samples]}}}}}
    """
    results_dict = {}

    # Default parameter settings - based on user requirements
    if datasets is None:
        datasets = ['stackexchange', 'dclm-privacy', 'wiki-fact']
    if prefix_lengths is None:
        prefix_lengths = [16, 32, 64]
    if generation_lengths is None:
        generation_lengths = [8, 16, 128]

    print(f"开始从基础目录加载数据: {results_base_dir}")
    print(f"目标generation lengths: {generation_lengths}")

    # Iterate over folders for each generation_length
    for gen_length in generation_lengths:
        gen_folder = f"exp1_generation_{gen_length}"
        gen_dir = os.path.join(results_base_dir, gen_folder)

        if not os.path.exists(gen_dir):
            print(f"警告: 文件夹 {gen_dir} 不存在，跳过generation_length={gen_length}")
            continue

        print(f"\n处理generation_length={gen_length}的文件夹: {gen_folder}")

        # Find all jsonl files in this folder
        pattern = os.path.join(gen_dir, "*.jsonl")
        result_files = glob.glob(pattern)

        if not result_files:
            print(f"警告: 在 {gen_dir} 中未找到结果文件")
            continue

        print(f"在 {gen_folder} 中找到 {len(result_files)} 个结果文件")

        for filepath in result_files:
            try:
                # Parse info from filename
                filename = os.path.basename(filepath)
                print(f"正在处理文件: {filename}")

                # Remove .jsonl suffix
                filename_parts = filename.replace('.jsonl', '').split('_')

                # Expected filename format: dataset_prefix{length}_{model_scale}_{model_type}_{num_samples}_samples.jsonl
                if len(filename_parts) >= 5:
                    dataset = filename_parts[0]
                    prefix_info = filename_parts[1]  # prefix{length}
                    file_model_scale = filename_parts[2]
                    model_type = filename_parts[3]

                    # Apply filters
                    if datasets and dataset not in datasets:
                        print(f"跳过数据集 {dataset} (不在目标列表中)")
                        continue

                    if model_scales and file_model_scale not in model_scales:
                        print(f"跳过模型规模 {file_model_scale} (不在目标列表中)")
                        continue

                    # Extract prefix length
                    try:
                        prefix_length = int(prefix_info.replace('prefix', ''))
                    except:
                        print(f"无法解析前缀长度: {prefix_info}")
                        continue

                    if prefix_lengths and prefix_length not in prefix_lengths:
                        print(f"跳过前缀长度 {prefix_length} (不在目标列表中)")
                        continue

                    # Load jsonl data
                    samples = []

                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line_idx, line in enumerate(f):
                            if line.strip():
                                try:
                                    sample = json.loads(line)
                                    samples.append(sample)

                                except json.JSONDecodeError as e:
                                    print(f"JSON解析错误 在文件 {filename} 第 {line_idx + 1} 行: {e}")
                                    continue

                    if not samples:
                        print(f"文件 {filename} 中没有有效样本")
                        continue

                    # Apply sample count cap
                    if max_samples and len(samples) > max_samples:
                        samples = samples[:max_samples]
                        print(f"样本数量限制为 {max_samples}")

                    # Organize data structure
                    if dataset not in results_dict:
                        results_dict[dataset] = {}
                    if file_model_scale not in results_dict[dataset]:
                        results_dict[dataset][file_model_scale] = {'base': {}, 'sft': {}}
                    if prefix_length not in results_dict[dataset][file_model_scale][model_type]:
                        results_dict[dataset][file_model_scale][model_type][prefix_length] = {}
                    if gen_length not in results_dict[dataset][file_model_scale][model_type][prefix_length]:
                        results_dict[dataset][file_model_scale][model_type][prefix_length][gen_length] = []

                    results_dict[dataset][file_model_scale][model_type][prefix_length][gen_length].extend(samples)
                    print(
                        f"✓ 加载 {dataset}-{file_model_scale}-{model_type}-prefix{prefix_length}-gen{gen_length}: {len(samples)} 条样本")

                else:
                    print(f"警告: 文件名格式不符合预期: {filename}")

            except Exception as e:
                print(f"加载文件 {filepath} 时出错: {e}")
                continue

    # Print loading summary
    total_configs = 0
    for dataset in results_dict:
        for model_scale in results_dict[dataset]:
            for model_type in ['base', 'sft']:
                if model_type in results_dict[dataset][model_scale]:
                    for prefix_length in results_dict[dataset][model_scale][model_type]:
                        total_configs += len(results_dict[dataset][model_scale][model_type][prefix_length])

    print(f"\n数据加载完成! 总共加载了 {total_configs} 个配置组合")
    return results_dict


def calculate_memorization_metrics_with_evaluator(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compute memorization metrics using the MemorizationMetrics class (memory-optimized).

    Args:
        results_dict: Generation results dictionary.

    Returns:
        metrics_df: DataFrame of memorization metrics.
    """
    # Initialize evaluator
    try:
        evaluator = MemorizationMetrics()
        print("✓ MemorizationMetrics评估器初始化成功")
    except Exception as e:
        print(f"❌ 评估器初始化失败: {e}")
        return pd.DataFrame()

    metrics_data = []

    for dataset in results_dict:
        for model_scale in results_dict[dataset]:
            for model_type in ['base', 'sft']:
                if model_type not in results_dict[dataset][model_scale]:
                    continue

                for prefix_length in results_dict[dataset][model_scale][model_type]:
                    for generation_length, samples in results_dict[dataset][model_scale][model_type][
                        prefix_length].items():
                        if not samples:
                            continue

                        print(
                            f"计算 {dataset}-{model_scale}-{model_type}-prefix{prefix_length}-gen{generation_length} 的记忆指标...")

                        try:
                            # Use the correct evaluator interface
                            metrics_results = evaluator.compute_all_metrics_from_data(samples)

                            # Extract metric values
                            metrics_entry = {
                                'dataset': dataset,
                                'model_type': model_type,
                                'model_scale': model_scale,
                                'prefix_length': prefix_length,
                                'generation_length': generation_length,
                                'sample_count': len(samples),
                            }

                            # Method 1: Exact match
                            if 'exact_match' in metrics_results:
                                metrics_entry['exact_match_rate'] = metrics_results['exact_match']['exact_match_rate']

                            # Method 2: ROUGE/BLEU
                            if 'rouge_bleu' in metrics_results:
                                rouge_bleu = metrics_results['rouge_bleu']
                                metrics_entry['rouge_1_f'] = rouge_bleu.get('rouge_1_f', 0.0)
                                metrics_entry['rouge_2_f'] = rouge_bleu.get('rouge_2_f', 0.0)
                                metrics_entry['rouge_l_f'] = rouge_bleu.get('rouge_l_f', 0.0)
                                metrics_entry['bleu_1'] = rouge_bleu.get('bleu_1', 0.0)
                                metrics_entry['bleu_2'] = rouge_bleu.get('bleu_2', 0.0)
                                metrics_entry['bleu_4'] = rouge_bleu.get('bleu_4', 0.0)

                            # Method 3: Edit distance
                            if 'edit_distance' in metrics_results:
                                edit_dist = metrics_results['edit_distance']
                                metrics_entry['token_edit_distance'] = edit_dist.get('token_edit_distance', 0.0)

                            # Method 5: Probability-related metrics
                            if 'likelihood' in metrics_results:
                                likelihood = metrics_results['likelihood']
                                metrics_entry['target_token_probability'] = likelihood.get('target_token_probability',
                                                                                           0.0)
                                metrics_entry['target_token_rank'] = likelihood.get('target_token_rank', float('inf'))
                                metrics_entry['target_in_top1_rate'] = likelihood.get('target_in_top1_rate', 0.0)
                                metrics_entry['target_in_top5_rate'] = likelihood.get('target_in_top5_rate', 0.0)
                            else:
                                # If likelihood data is missing, set defaults
                                metrics_entry['target_token_probability'] = 0.0
                                metrics_entry['target_token_rank'] = float('inf')
                                metrics_entry['target_in_top1_rate'] = 0.0
                                metrics_entry['target_in_top5_rate'] = 0.0

                            metrics_data.append(metrics_entry)
                            print(f"✓ 完成计算，处理 {len(samples)} 个样本")

                        except Exception as e:
                            print(f"❌ 计算指标时出错: {e}")
                            continue

                        # Memory cleanup
                        del samples

    if not metrics_data:
        print("❌ 没有成功计算任何指标")
        return pd.DataFrame()

    return pd.DataFrame(metrics_data)


def generate_delta_tables(metrics_df: pd.DataFrame,
                          output_dir: str,
                          prefix_lengths: List[int],
                          generation_lengths: List[int]):
    """
    Generate delta tables for SFT relative to Base (SFT - Base).

    Args:
        metrics_df: DataFrame containing all metrics.
        output_dir: Output directory.
        prefix_lengths: List of prefix lengths.
        generation_lengths: List of generation lengths.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Dataset name mapping
    dataset_mapping = {
        'stackexchange': 'STACKEXCHANGE',
        'dclm-privacy': 'DCLM-PRIVACY',
        'wiki-fact': 'WIKI-FACT'
    }

    # Metrics to compute deltas for
    metrics_to_generate = [
        ('exact_match_rate', 'Exact Match Rate'),
        ('rouge_1_f', 'ROUGE-1 F-score'),
        ('rouge_2_f', 'ROUGE-2 F-score'),
        ('rouge_l_f', 'ROUGE-L F-score'),
        ('bleu_1', 'BLEU-1'),
        ('bleu_2', 'BLEU-2'),
        ('bleu_4', 'BLEU-4'),
        ('token_edit_distance', 'Token Edit Distance'),
        ('target_token_probability', 'Target Token Probability')
    ]

    # Collect all delta tables
    all_delta_tables = []

    # Unique datasets and models
    datasets = sorted(metrics_df['dataset'].unique())
    model_scales = sort_model_scales(metrics_df['model_scale'].unique())

    print(f"\n=== 生成Delta表格 (SFT - Base) ===")
    print(f"数据集: {datasets}")
    print(f"模型规模: {model_scales}")

    # Generate one large consolidated table per metric
    for metric_col, metric_name in metrics_to_generate:
        if metric_col not in metrics_df.columns:
            print(f"警告: 指标 {metric_col} 不在数据中")
            continue

        print(f"生成 {metric_name} 的Delta表格...")

        # Build row groups: model x prefix_length
        row_groups = []
        for model_scale in model_scales:
            for prefix_len in prefix_lengths:
                row_groups.append((model_scale, prefix_len))

        # Build column groups: dataset x generation_length
        col_groups = []
        for dataset in datasets:
            for gen_len in generation_lengths:
                col_groups.append((dataset, gen_len))

        # Build table data
        table_data = []
        for model_scale, prefix_len in row_groups:
            row_data = []
            for dataset, gen_len in col_groups:
                # Filter data for the current condition
                condition_mask = (
                    (metrics_df['dataset'] == dataset) &
                    (metrics_df['model_scale'] == model_scale) &
                    (metrics_df['prefix_length'] == prefix_len) &
                    (metrics_df['generation_length'] == gen_len)
                )

                # Find base and sft values
                base_mask = condition_mask & (metrics_df['model_type'] == 'base')
                sft_mask = condition_mask & (metrics_df['model_type'] == 'sft')

                base_value = None
                sft_value = None

                if base_mask.sum() > 0:
                    base_value = metrics_df.loc[base_mask, metric_col].iloc[0]

                if sft_mask.sum() > 0:
                    sft_value = metrics_df.loc[sft_mask, metric_col].iloc[0]

                # Compute delta value (SFT - Base)
                if (base_value is not None and sft_value is not None and
                        not pd.isna(base_value) and not pd.isna(sft_value) and
                        base_value != float('inf') and sft_value != float('inf') and
                        base_value != float('-inf') and sft_value != float('-inf')):

                    delta_value = sft_value - base_value

                    # Formatting and sign logic depends on metric type
                    if metric_col == 'token_edit_distance':
                        # Smaller edit distance is better; negative delta indicates improvement
                        formatted_value = f"{delta_value:.1f}"
                        if delta_value < 0:
                            formatted_value = f"\\textcolor{{green}}{{{formatted_value}}}"
                        elif delta_value > 0:
                            formatted_value = f"\\textcolor{{red}}{{{formatted_value}}}"
                    else:
                        # For other metrics, larger is better; positive delta indicates improvement
                        formatted_value = f"{delta_value:.3f}"
                        if delta_value > 0:
                            formatted_value = f"\\textcolor{{green}}{{+{formatted_value}}}"
                        elif delta_value < 0:
                            formatted_value = f"\\textcolor{{red}}{{{formatted_value}}}"

                    row_data.append(formatted_value)
                else:
                    row_data.append('N/A')

            table_data.append(row_data)

        # Generate LaTeX table
        latex_table = generate_single_latex_table(
            table_data,
            row_groups,
            col_groups,
            datasets,
            generation_lengths,
            metric_name,
            dataset_mapping,
            model_scales,
            is_delta_table=True
        )

        all_delta_tables.append(latex_table)
        print(f"✓ {metric_name} Delta表格已生成")

    # Save delta tables to a separate file
    prefix_str = '_'.join(map(str, prefix_lengths))
    gen_str = '_'.join(map(str, generation_lengths))
    delta_output_file = os.path.join(output_dir, f'memorization_delta_tables_prefix{prefix_str}_gen{gen_str}.tex')

    with open(delta_output_file, 'w', encoding='utf-8') as f:
        # Write LaTeX document header
        f.write("% Memorization Metrics Delta Tables (SFT - Base)\n")
        f.write("% Generated automatically\n")
        f.write(f"% Prefix lengths: {prefix_lengths}\n")
        f.write(f"% Generation lengths: {generation_lengths}\n")
        f.write("% Requires booktabs, xcolor and multirow packages: \\usepackage{booktabs} \\usepackage{xcolor} \\usepackage{multirow}\n\n")

        for i, table in enumerate(all_delta_tables):
            f.write(table)
            if i < len(all_delta_tables) - 1:
                f.write("\n\n\\clearpage\n\n")

    print(f"\n🎯 Delta表格已保存到: {delta_output_file}")
    print(f"总共生成了 {len(all_delta_tables)} 个Delta表格")

    # Print a preview of the first delta table
    if all_delta_tables:
        print(f"\n=== Delta表格预览 ===")
        print(all_delta_tables[0])

    return delta_output_file

def generate_single_latex_table(table_data: List[List[str]],
                                row_groups: List[tuple],
                                col_groups: List[tuple],
                                datasets: List[str],
                                generation_lengths: List[int],
                                table_title: str,
                                dataset_mapping: dict,
                                model_scales: List[str] = None,
                                is_delta_table: bool = False) -> str:
    """
    Generate a single LaTeX table, supporting model group display and delta table formatting.

    Args:
        table_data: Table cell data.
        row_groups: Row groups [(model_scale, prefix_length), ...].
        col_groups: Column groups [(dataset, generation_length), ...].
        datasets: List of datasets.
        generation_lengths: List of generation lengths.
        table_title: Table caption/title.
        dataset_mapping: Dataset name mapping.
        model_scales: List of model scales (used for inserting group separators).
        is_delta_table: Whether this is a delta table.

    Returns:
        latex_code: Generated LaTeX code.
    """
    num_datasets = len(datasets)
    num_gen_lengths = len(generation_lengths)
    total_cols = len(col_groups)

    # Begin table
    latex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{table_title}}}",
        f"\\begin{{tabular}}{{ll|{'c' * total_cols}}}",
        "\\toprule"
    ]

    # Header row 1: dataset names (spanning multiple columns)
    header1_parts = ["\\multirow{2}{*}{Model}", "\\multirow{2}{*}{Prefix L}"]
    for dataset in datasets:
        dataset_name = dataset_mapping.get(dataset, dataset)
        header1_parts.append(f"\\multicolumn{{{num_gen_lengths}}}{{c}}{{{dataset_name}}}")
    header1 = " & ".join(header1_parts) + " \\\\"
    latex_lines.append(header1)

    # Header row 2: Generation L under each dataset
    header2_parts = ["", ""]  # Empty cells for Model and Prefix L
    for _ in datasets:
        for gen_len in generation_lengths:
            header2_parts.append(str(gen_len))
    header2 = " & ".join(header2_parts) + " \\\\"
    latex_lines.append(header2)
    latex_lines.append("\\midrule")

    # Fill table body
    current_model = None
    for i, ((model_scale, prefix_len), row_data) in enumerate(zip(row_groups, table_data)):
        # If this is a new model group, show the model name with multirow
        if model_scale != current_model:
            # Number of rows for this model (equals number of prefix lengths)
            model_row_count = sum(1 for m, p in row_groups if m == model_scale)

            # First row: display model name
            row_parts = [
                f"\\multirow{{{model_row_count}}}{{*}}{{{model_scale}}}",
                str(prefix_len)
            ] + row_data

            current_model = model_scale
        else:
            # Subsequent rows: do not display model name
            row_parts = ["", str(prefix_len)] + row_data

        row_str = " & ".join(row_parts) + " \\\\"
        latex_lines.append(row_str)

        # Add a separator line between model groups
        next_idx = i + 1
        if next_idx < len(row_groups):
            next_model, _ = row_groups[next_idx]
            if next_model != current_model:
                latex_lines.append("\\midrule")

    # End table
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\label{{tab:{table_title.lower().replace(' ', '_').replace('-', '_').replace(':', '').replace('(', '').replace(')', '')}}}",
        "\\end{table*}"
    ])

    return "\n".join(latex_lines)



def main():
    """Main entry: parse arguments and run the table generation pipeline."""

    parser = argparse.ArgumentParser(description='生成记忆指标的LaTeX表格')

    parser.add_argument('--results_base_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/results',
                        help='结果基础目录路径，包含exp1_generation_X子文件夹')
    parser.add_argument('--model_scales', type=str, nargs='+',
                        default=["1B", "7B", "13B", "32B"],
                        help='要分析的模型规模列表，如 ["1B", "7B", "13B", "32B"]')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['stackexchange', 'dclm-privacy', 'wiki-fact'],
                        help='要分析的数据集列表')
    parser.add_argument('--prefix_lengths', type=int, nargs='+',
                        default=[16, 32, 64],
                        help='要分析的前缀长度列表')
    parser.add_argument('--generation_lengths', type=int, nargs='+',
                        default=[8, 16],
                        help='要分析的生成长度列表(continuation L)')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='每种条件下的最大样本数，设置较小值以适应2G内存限制')
    parser.add_argument('--output_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/results/exp1_mem_score',
                        help='LaTeX表格输出目录')

    args = parser.parse_args()

    print("=" * 80)
    print("开始生成记忆指标LaTeX表格...")
    print("=" * 80)
    print(f"结果基础目录: {args.results_base_dir}")
    print(f"模型规模: {args.model_scales}")
    print(f"数据集: {args.datasets}")
    print(f"前缀长度: {args.prefix_lengths}")
    print(f"生成长度: {args.generation_lengths}")
    print(f"最大样本数: {args.max_samples}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 80)

    # Load generation results
    print("\n步骤1: 加载生成结果...")
    results_dict = load_generation_results_memory_optimized(
        args.results_base_dir,
        args.model_scales,
        args.datasets,
        args.prefix_lengths,
        args.generation_lengths,
        args.max_samples
    )

    if not results_dict:
        print("❌ 错误: 未能加载任何生成结果")
        return

    # Compute memorization metrics
    print("\n步骤2: 计算记忆指标...")
    metrics_df = calculate_memorization_metrics_with_evaluator(results_dict)

    if len(metrics_df) == 0:
        print("❌ 错误: 无法计算记忆指标")
        return

    print(f"✓ 计算完成，共 {len(metrics_df)} 条记录")
    print("\n指标概览:")
    if len(metrics_df) > 0:
        preview_cols = ['dataset', 'model_scale', 'model_type', 'prefix_length',
                        'generation_length', 'exact_match_rate', 'rouge_1_f', 'bleu_1']
        available_cols = [col for col in preview_cols if col in metrics_df.columns]
        print(metrics_df[available_cols].head(10))

    # Generate and save LaTeX tables
    print("\n步骤3: 生成LaTeX表格...")

    # First generate delta tables (SFT - Base)
    print("\n🎯 步骤3.1: 生成Delta表格 (SFT - Base)...")
    delta_file = generate_delta_tables(metrics_df, args.output_dir, args.prefix_lengths, args.generation_lengths)



    print("\n" + "=" * 80)
    print("🎉 分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
