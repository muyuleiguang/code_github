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
        初始化评估指标

        Args:
            tokenizer_name: tokenizer模型名称（可选）
            sentence_model_name: 句子embedding模型名称
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
        第一种：精确匹配率 (Exact Match Rate)

        模型生成的内容与训练数据原文完全一致的比例。这是最严格的指标。

        Args:
            generated_tokens: 生成的token列表
            reference_tokens: 参考token列表

        Returns:
            精确匹配指标
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
        第二种：ROUGE / BLEU 分数

        用于衡量生成文本和参考文本之间的n-gram重叠度。
        可以捕捉到近似记忆（near-verbatim memorization）。

        Args:
            generated_texts: 生成的文本列表
            reference_texts: 参考文本列表
            generated_tokens: 生成的token列表（可选，用于token级BLEU）
            reference_tokens: 参考token列表（可选，用于token级BLEU）

        Returns:
            ROUGE和BLEU指标
        """
        results = {}

        # ROUGE scores (基于文本)
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

        # BLEU scores (基于token，更精确)
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

                # 将token ID转换为字符串用于BLEU计算
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
        第三种：编辑距离 (Edit Distance)

        生成文本需要经过多少次增、删、改才能变成原文。
        距离越小，记忆程度越高。主要使用Token-level Edit Distance。

        Args:
            generated_tokens: 生成的token列表
            reference_tokens: 参考token列表
            generated_texts: 生成的文本列表（可选）
            reference_texts: 参考文本列表（可选）

        Returns:
            编辑距离指标
        """
        assert len(generated_tokens) == len(reference_tokens)

        token_distances = []
        normalized_token_distances = []

        # Token级编辑距离（主要指标）
        for gen_tokens, ref_tokens in zip(generated_tokens, reference_tokens):
            token_dist = editdistance.eval(gen_tokens, ref_tokens)
            token_distances.append(token_dist)

            # 归一化token编辑距离
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

        # 字符级编辑距离（如果提供了文本）
        if generated_texts and reference_texts:
            char_distances = []
            normalized_char_distances = []

            for gen_text, ref_text in zip(generated_texts, reference_texts):
                char_dist = editdistance.eval(gen_text, ref_text)
                char_distances.append(char_dist)

                # 归一化字符编辑距离
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
        第四种：语义相似度

        使用SentenceBERT等计算embedding相似度

        Args:
            generated_texts: 生成的文本列表
            reference_texts: 参考文本列表

        Returns:
            语义相似度指标
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
        第五种：Likelihood, PPL, loss, logits

        基于模型输出概率计算记忆相关指标

        Args:
            top_tokens_list: 每个样本的每一步top-k token概率信息
                格式: [sample][step][top_k_tokens]
                每个token_info包含: {'token_id': int, 'probability': float, 'rank': int}
            reference_tokens: 参考token列表
            logits: 完整的logits张量（可选，如果有的话更精确）

        Returns:
            likelihood, perplexity, loss相关指标
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

                # 查找目标token在top-k中的位置和概率
                target_found = False

                for rank, token_info in enumerate(step_top_tokens):
                    token_id = token_info.get('token_id')
                    prob = token_info.get('probability', 0.0)

                    if token_id == target_token:
                        target_probs.append(prob)
                        target_ranks.append(rank + 1)  # rank从1开始
                        target_found = True

                        # 计算log likelihood
                        if prob > 0:
                            log_prob = np.log(prob)
                            sample_log_likelihood += log_prob

                            # 计算cross-entropy loss
                            loss = -log_prob
                            losses.append(loss)
                        else:
                            sample_log_likelihood += -100  # 避免log(0)
                            losses.append(100)

                        # 统计top-k命中率
                        if rank == 0:  # top-1
                            top1_hits += 1
                        if rank < 3:  # top-3
                            top3_hits += 1
                        if rank < 5:  # top-5
                            top5_hits += 1
                        break

                if not target_found:
                    # 目标token不在top-k中，使用很小的概率
                    target_probs.append(1e-10)
                    target_ranks.append(float('inf'))
                    sample_log_likelihood += -100
                    losses.append(100)

            if sample_positions > 0:
                log_likelihoods.append(sample_log_likelihood / sample_positions)

        # 计算平均指标
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
        从样本数据计算所有5种评估指标

        Args:
            samples: 样本列表，每个样本包含generated_tokens, original_continuation_tokens等字段

        Returns:
            包含所有指标结果的字典
        """
        # 提取数据
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

        # 第一种：精确匹配率
        if generated_tokens and reference_tokens:
            results["exact_match"] = self.exact_match_rate(generated_tokens, reference_tokens)

        # 第二种：ROUGE/BLEU分数
        if generated_texts and reference_texts:
            results["rouge_bleu"] = self.rouge_bleu_scores(
                generated_texts, reference_texts, generated_tokens, reference_tokens
            )

        # 第三种：编辑距离
        if generated_tokens and reference_tokens:
            results["edit_distance"] = self.edit_distance_metrics(
                generated_tokens, reference_tokens, generated_texts, reference_texts
            )

        # 第四种：语义相似度
        if generated_texts and reference_texts:
            results["semantic"] = self.semantic_similarity(generated_texts, reference_texts)

        # 第五种：Likelihood, PPL, loss
        if top_tokens_list and reference_tokens:
            results["likelihood"] = self.likelihood_ppl_loss_metrics(top_tokens_list, reference_tokens)

        return results


def sort_model_scales(model_scales):
    """
    按照模型规模数值大小排序，如 1B < 7B < 13B < 32B

    Args:
        model_scales: 模型规模列表

    Returns:
        sorted_scales: 排序后的模型规模列表
    """

    def extract_scale_value(scale_str):
        """提取模型规模的数值部分用于排序"""
        try:
            # 移除末尾的单位（B、M等）
            if scale_str.endswith('B'):
                return float(scale_str[:-1])
            elif scale_str.endswith('M'):
                return float(scale_str[:-1]) / 1000  # 转换为B单位
            else:
                # 如果没有单位，直接当作数字处理
                return float(scale_str)
        except:
            # 如果解析失败，返回一个大数值，排在最后
            return float('inf')

    return sorted(model_scales, key=extract_scale_value)


def load_generation_results_memory_optimized(results_base_dir: str,
                                             model_scales: List[str],
                                             datasets: List[str] = None,
                                             prefix_lengths: List[int] = None,
                                             generation_lengths: List[int] = None,
                                             max_samples: int = None) -> Dict[str, Dict]:
    """
    内存优化版本：分批加载生成结果文件，减少内存占用

    Args:
        results_base_dir: 结果基础目录路径，包含exp1_generation_X子文件夹
        model_scales: 模型规模列表 (如 ["1B", "7B", "13B"])
        datasets: 要加载的数据集列表，None表示加载所有
        prefix_lengths: 要加载的前缀长度列表，None表示加载所有
        generation_lengths: 要加载的生成长度列表，None表示加载所有
        max_samples: 每个条件下的最大样本数，默认50减少内存使用

    Returns:
        results_dict: 按数据集、模型规模、模型类型、前缀长度和生成长度组织的结果
    """
    results_dict = {}

    # 默认参数设置 - 内存优化
    if datasets is None:
        datasets = ['stackexchange', 'dclm-privacy', 'wiki-fact']
    if prefix_lengths is None:
        prefix_lengths = [16, 32, 64]
    if generation_lengths is None:
        generation_lengths = [8, 16, 128]
    if max_samples is None:
        max_samples = 50  # 内存优化：默认限制样本数

    print(f"内存优化模式: 每个配置最多加载 {max_samples} 个样本")
    print(f"开始从基础目录加载数据: {results_base_dir}")

    # 遍历每个generation_length对应的文件夹
    for gen_length in generation_lengths:
        gen_folder = f"exp1_generation_{gen_length}"
        gen_dir = os.path.join(results_base_dir, gen_folder)

        if not os.path.exists(gen_dir):
            print(f"警告: 文件夹 {gen_dir} 不存在，跳过generation_length={gen_length}")
            continue

        print(f"\n处理generation_length={gen_length}的文件夹: {gen_folder}")

        # 搜索该文件夹内的所有jsonl文件
        pattern = os.path.join(gen_dir, "*.jsonl")
        result_files = glob.glob(pattern)

        if not result_files:
            print(f"警告: 在 {gen_dir} 中未找到结果文件")
            continue

        print(f"在 {gen_folder} 中找到 {len(result_files)} 个结果文件")

        for filepath in result_files:
            try:
                # 从文件名解析信息
                filename = os.path.basename(filepath)
                print(f"正在处理文件: {filename}")

                filename_parts = filename.replace('.jsonl', '').split('_')

                if len(filename_parts) >= 5:
                    dataset = filename_parts[0]
                    prefix_info = filename_parts[1]
                    file_model_scale = filename_parts[2]
                    model_type = filename_parts[3]

                    # 应用过滤条件
                    if datasets and dataset not in datasets:
                        continue
                    if model_scales and file_model_scale not in model_scales:
                        continue

                    try:
                        prefix_length = int(prefix_info.replace('prefix', ''))
                    except:
                        continue

                    if prefix_lengths and prefix_length not in prefix_lengths:
                        continue

                    # 内存优化：分批读取文件
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

                    # 组织数据结构
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
    加载多个数据集和模型规模的生成结果文件
    支持按generation_length分文件夹存储的结构

    Args:
        results_base_dir: 结果基础目录路径，包含exp1_generation_X子文件夹
        model_scales: 模型规模列表 (如 ["1B", "7B", "13B", "32B"])
        datasets: 要加载的数据集列表，None表示加载所有
        prefix_lengths: 要加载的前缀长度列表，None表示加载所有
        generation_lengths: 要加载的生成长度列表(continuation L)，None表示加载所有
        max_samples: 每个条件下的最大样本数，None表示加载所有

    Returns:
        results_dict: 按数据集、模型规模、模型类型、前缀长度和生成长度组织的结果
        格式: {dataset: {model_scale: {model_type: {prefix_length: {generation_length: [samples]}}}}}
    """
    results_dict = {}

    # 默认参数设置 - 基于用户需求
    if datasets is None:
        datasets = ['stackexchange', 'dclm-privacy', 'wiki-fact']
    if prefix_lengths is None:
        prefix_lengths = [16, 32, 64]
    if generation_lengths is None:
        generation_lengths = [8, 16, 128]

    print(f"开始从基础目录加载数据: {results_base_dir}")
    print(f"目标generation lengths: {generation_lengths}")

    # 遍历每个generation_length对应的文件夹
    for gen_length in generation_lengths:
        gen_folder = f"exp1_generation_{gen_length}"
        gen_dir = os.path.join(results_base_dir, gen_folder)

        if not os.path.exists(gen_dir):
            print(f"警告: 文件夹 {gen_dir} 不存在，跳过generation_length={gen_length}")
            continue

        print(f"\n处理generation_length={gen_length}的文件夹: {gen_folder}")

        # 搜索该文件夹内的所有jsonl文件
        pattern = os.path.join(gen_dir, "*.jsonl")
        result_files = glob.glob(pattern)

        if not result_files:
            print(f"警告: 在 {gen_dir} 中未找到结果文件")
            continue

        print(f"在 {gen_folder} 中找到 {len(result_files)} 个结果文件")

        for filepath in result_files:
            try:
                # 从文件名解析信息
                filename = os.path.basename(filepath)
                print(f"正在处理文件: {filename}")

                # 移除.jsonl后缀
                filename_parts = filename.replace('.jsonl', '').split('_')

                # 解析文件名格式: dataset_prefix{length}_{model_scale}_{model_type}_{num_samples}_samples.jsonl
                if len(filename_parts) >= 5:
                    dataset = filename_parts[0]
                    prefix_info = filename_parts[1]  # prefix{length}
                    file_model_scale = filename_parts[2]
                    model_type = filename_parts[3]

                    # 应用过滤条件
                    if datasets and dataset not in datasets:
                        print(f"跳过数据集 {dataset} (不在目标列表中)")
                        continue

                    if model_scales and file_model_scale not in model_scales:
                        print(f"跳过模型规模 {file_model_scale} (不在目标列表中)")
                        continue

                    # 提取前缀长度
                    try:
                        prefix_length = int(prefix_info.replace('prefix', ''))
                    except:
                        print(f"无法解析前缀长度: {prefix_info}")
                        continue

                    if prefix_lengths and prefix_length not in prefix_lengths:
                        print(f"跳过前缀长度 {prefix_length} (不在目标列表中)")
                        continue

                    # 加载jsonl数据
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

                    # 应用样本数过滤
                    if max_samples and len(samples) > max_samples:
                        samples = samples[:max_samples]
                        print(f"样本数量限制为 {max_samples}")

                    # 组织数据结构
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

    # 打印加载摘要
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
    使用MemorizationMetrics类计算记忆指标（内存优化版本）

    Args:
        results_dict: 生成结果字典

    Returns:
        metrics_df: 记忆指标的DataFrame
    """
    # 初始化评估器
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
                            # 使用正确的评估器接口
                            metrics_results = evaluator.compute_all_metrics_from_data(samples)

                            # 提取各种指标数据
                            metrics_entry = {
                                'dataset': dataset,
                                'model_type': model_type,
                                'model_scale': model_scale,
                                'prefix_length': prefix_length,
                                'generation_length': generation_length,
                                'sample_count': len(samples),
                            }

                            # 第一种：精确匹配
                            if 'exact_match' in metrics_results:
                                metrics_entry['exact_match_rate'] = metrics_results['exact_match']['exact_match_rate']

                            # 第二种：ROUGE/BLEU指标
                            if 'rouge_bleu' in metrics_results:
                                rouge_bleu = metrics_results['rouge_bleu']
                                metrics_entry['rouge_1_f'] = rouge_bleu.get('rouge_1_f', 0.0)
                                metrics_entry['rouge_2_f'] = rouge_bleu.get('rouge_2_f', 0.0)
                                metrics_entry['rouge_l_f'] = rouge_bleu.get('rouge_l_f', 0.0)
                                metrics_entry['bleu_1'] = rouge_bleu.get('bleu_1', 0.0)
                                metrics_entry['bleu_2'] = rouge_bleu.get('bleu_2', 0.0)
                                metrics_entry['bleu_4'] = rouge_bleu.get('bleu_4', 0.0)

                            # 第三种：编辑距离
                            if 'edit_distance' in metrics_results:
                                edit_dist = metrics_results['edit_distance']
                                metrics_entry['token_edit_distance'] = edit_dist.get('token_edit_distance', 0.0)

                            # 第五种：概率相关指标
                            if 'likelihood' in metrics_results:
                                likelihood = metrics_results['likelihood']
                                metrics_entry['target_token_probability'] = likelihood.get('target_token_probability',
                                                                                           0.0)
                                metrics_entry['target_token_rank'] = likelihood.get('target_token_rank', float('inf'))
                                metrics_entry['target_in_top1_rate'] = likelihood.get('target_in_top1_rate', 0.0)
                                metrics_entry['target_in_top5_rate'] = likelihood.get('target_in_top5_rate', 0.0)
                            else:
                                # 如果没有likelihood数据，设置默认值
                                metrics_entry['target_token_probability'] = 0.0
                                metrics_entry['target_token_rank'] = float('inf')
                                metrics_entry['target_in_top1_rate'] = 0.0
                                metrics_entry['target_in_top5_rate'] = 0.0

                            metrics_data.append(metrics_entry)
                            print(f"✓ 完成计算，处理 {len(samples)} 个样本")

                        except Exception as e:
                            print(f"❌ 计算指标时出错: {e}")
                            continue

                        # 内存清理
                        del samples

    if not metrics_data:
        print("❌ 没有成功计算任何指标")
        return pd.DataFrame()

    return pd.DataFrame(metrics_data)


def generate_latex_tables(metrics_df: pd.DataFrame,
                          output_dir: str,
                          prefix_lengths: List[int],
                          generation_lengths: List[int]):
    """
    生成LaTeX表格，每个评估指标生成一个表格
    按照用户需求的格式生成表格

    Args:
        metrics_df: 包含所有指标的DataFrame
        output_dir: 输出目录
        prefix_lengths: 前缀长度列表
        generation_lengths: 生成长度列表
    """

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 数据集名称映射（中英文对照）
    dataset_mapping = {
        'stackexchange': 'STACKEXCHANGE',
        'dclm-privacy': 'DCLM-PRIVACY',
        'wiki-fact': 'WIKI-FACT'
    }

    # 需要生成的指标及其显示名称（基于MemorizationMetrics的输出）
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

    # 收集所有latex表格
    all_latex_tables = []

    # 获取唯一的数据集和模型
    datasets = sorted(metrics_df['dataset'].unique())
    model_scales = sort_model_scales(metrics_df['model_scale'].unique())

    print(f"发现数据集: {datasets}")
    print(f"发现模型规模: {model_scales}")

    # 为每个prefix_length和generation_length组合生成表格
    for prefix_length in prefix_lengths:
        for generation_length in generation_lengths:
            print(f"\n=== 生成 prefix_length={prefix_length}, generation_length={generation_length} 的表格 ===")

            # 筛选当前条件的数据
            condition_df = metrics_df[
                (metrics_df['prefix_length'] == prefix_length) &
                (metrics_df['generation_length'] == generation_length)
                ]

            if len(condition_df) == 0:
                print(f"警告: 没有找到 prefix_length={prefix_length}, generation_length={generation_length} 的数据")
                continue

            print(f"找到 {len(condition_df)} 条记录")

            # 为每个指标生成表格
            for metric_col, metric_name in metrics_to_generate:
                if metric_col not in condition_df.columns:
                    print(f"警告: 指标 {metric_col} 不在数据中")
                    continue

                print(f"生成 {metric_name} 的表格 (prefix={prefix_length}, gen={generation_length})...")

                # 创建表格数据
                table_data = []
                row_labels = []

                for model_scale in model_scales:
                    for model_type in ['base', 'sft']:
                        # 创建行标签：模型规模 + 模型类型
                        if model_type == 'base':
                            model_label = f"{model_scale}"
                        else:
                            model_label = f"{model_scale}"  # SFT行
                        row_labels.append(model_label)

                        row_data = []
                        for dataset in datasets:
                            # 查找对应的值
                            mask = (condition_df['dataset'] == dataset) & \
                                   (condition_df['model_scale'] == model_scale) & \
                                   (condition_df['model_type'] == model_type)

                            if mask.sum() > 0:
                                value = condition_df.loc[mask, metric_col].iloc[0]
                                if pd.isna(value) or value == float('inf') or value == float('-inf'):
                                    row_data.append('N/A')
                                else:
                                    # 根据指标类型决定格式
                                    if metric_col == 'token_edit_distance':
                                        row_data.append(f"{value:.1f}")
                                    else:
                                        row_data.append(f"{value:.3f}")
                            else:
                                row_data.append('N/A')

                        table_data.append(row_data)

                # 生成latex表格，标题包含prefix_length和generation_length信息
                table_title = f"{metric_name} (Prefix L: {prefix_length}, Generation L: {generation_length})"
                latex_table = generate_single_latex_table(
                    table_data,
                    row_labels,
                    [dataset_mapping.get(d, d) for d in datasets],
                    table_title,
                    model_scales  # 传入模型规模用于分组
                )

                all_latex_tables.append(latex_table)
                print(f"✓ {metric_name} 表格已生成")

    # 保存所有表格到一个文件
    prefix_str = '_'.join(map(str, prefix_lengths))
    gen_str = '_'.join(map(str, generation_lengths))
    output_file = os.path.join(output_dir, f'memorization_metrics_latex_prefix{prefix_str}_gen{gen_str}.tex')

    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入LaTeX文档头部
        f.write("% Memorization Metrics LaTeX Tables\n")
        f.write("% Generated automatically\n")
        f.write(f"% Prefix lengths: {prefix_lengths}\n")
        f.write(f"% Generation lengths: {generation_lengths}\n")
        f.write("% Requires booktabs package: \\usepackage{booktabs}\n\n")

        for i, table in enumerate(all_latex_tables):
            f.write(table)
            if i < len(all_latex_tables) - 1:
                f.write("\n\n\\clearpage\n\n")

    print(f"\n所有latex表格已保存到: {output_file}")
    print(f"总共生成了 {len(all_latex_tables)} 个表格")

    # 也生成一个总结的CSV文件
    summary_file = os.path.join(output_dir, f'memorization_metrics_summary_prefix{prefix_str}_gen{gen_str}.csv')
    metrics_df.to_csv(summary_file, index=False, encoding='utf-8')
    print(f"指标总结CSV已保存到: {summary_file}")

    # 打印表格内容到控制台
    print("\n=== 表格内容预览 ===")
    for i, table in enumerate(all_latex_tables[:2]):  # 只显示前两个表格避免输出过长
        print(f"\n表格 {i + 1}:")
        print(table)
        if i >= 1:
            print(f"\n... (共 {len(all_latex_tables)} 个表格，完整内容请查看文件)")
            break


def generate_single_latex_table(table_data: List[List[str]],
                                row_labels: List[str],
                                col_labels: List[str],
                                table_title: str,
                                model_scales: List[str] = None) -> str:
    """
    生成单个latex表格，支持模型分组显示

    Args:
        table_data: 表格数据
        row_labels: 行标签
        col_labels: 列标签
        table_title: 表格标题
        model_scales: 模型规模列表，用于添加分组线

    Returns:
        latex_code: 生成的latex代码
    """
    num_cols = len(col_labels)

    # 开始表格
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{table_title}}}",
        f"\\begin{{tabular}}{{l{'c' * num_cols}}}",
        "\\toprule"
    ]

    # 表头
    header = "Model & " + " & ".join(col_labels) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")

    # 表格数据，每两行（base和sft）为一组
    for i, (row_label, row_data) in enumerate(zip(row_labels, table_data)):
        # 如果是SFT行，在行标签前添加缩进
        if i % 2 == 1:  # SFT行
            row_str = "~~~" + row_label + " & " + " & ".join(row_data) + " \\\\"
        else:  # Base行
            row_str = row_label + " & " + " & ".join(row_data) + " \\\\"

        latex_lines.append(row_str)

        # 每个模型的base和sft之间添加小分隔
        if i % 2 == 1 and i < len(row_labels) - 1:
            latex_lines.append("\\addlinespace[0.1em]")

    # 结束表格
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\label{{tab:{table_title.lower().replace(' ', '_').replace('-', '_').replace(':', '').replace('(', '').replace(')', '')}}}",
        "\\end{table}"
    ])

    return "\n".join(latex_lines)


def main():
    """主函数，解析参数并执行表格生成流程"""

    parser = argparse.ArgumentParser(description='生成记忆指标的LaTeX表格')

    parser.add_argument('--results_base_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/results',
                        help='结果基础目录路径，包含exp1_generation_X子文件夹')
    parser.add_argument('--model_scales', type=str, nargs='+',
                        default=['1B', '7B', '13B', '32B'],
                        help='要分析的模型规模列表，如 ["1B", "7B", "13B", "32B"]')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['stackexchange', 'dclm-privacy', 'wiki-fact'],
                        help='要分析的数据集列表')
    parser.add_argument('--prefix_lengths', type=int, nargs='+',
                        default=[16, 32, 64],
                        help='要分析的前缀长度列表')
    parser.add_argument('--generation_lengths', type=int, nargs='+',
                        default=[8, 16, 128],
                        help='要分析的生成长度列表(continuation L)')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='每种条件下的最大样本数，设置较小值以适应2G内存限制')
    parser.add_argument('--output_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/results/latex_tables',
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

    # 加载生成结果
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

    # 计算记忆指标
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

    # 生成并保存latex表格
    print("\n步骤3: 生成LaTeX表格...")
    generate_latex_tables(metrics_df, args.output_dir, args.prefix_lengths, args.generation_lengths)

    print("\n" + "=" * 80)
    print("🎉 分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()