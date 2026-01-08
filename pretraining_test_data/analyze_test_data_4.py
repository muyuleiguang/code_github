"""
Statistical analysis for the final test dataset
"""
import json
import os
from typing import List, Dict
import re

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import argparse
import pandas as pd
from scipy import stats


class TestDataAnalyzer:
    def __init__(self):
        """Initialize the analyzer"""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    def load_data(self, file_path: str) -> List[Dict]:
        """Load data"""
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def analyze_selection_distribution(self, data: List[Dict]) -> Dict:
        """
        Analyze selection-reason distribution

        Why analyze:
        - Understand coverage across different data types
        - Ensure the effectiveness of the selection strategy
        """
        selection_reason_counts = Counter()
        selection_scores = defaultdict(list)
        word_counts = defaultdict(list)

        for item in data:
            reason = item.get("selection_reason", "unknown")
            score = item.get("selection_score", 0)
            word_count = item.get("word_count", 0)

            selection_reason_counts[reason] += 1
            selection_scores[reason].append(score)
            word_counts[reason].append(word_count)

        # Compute statistics for each reason
        reason_stats = {}
        for reason in selection_reason_counts:
            reason_stats[reason] = {
                "count": selection_reason_counts[reason],
                "avg_score": np.mean(selection_scores[reason]),
                "std_score": np.std(selection_scores[reason]),
                "avg_word_count": np.mean(word_counts[reason]),
                "std_word_count": np.std(word_counts[reason])
            }

        return {
            "selection_reason_counts": dict(selection_reason_counts),
            "reason_stats": reason_stats,
            "all_scores": {k: v for k, v in selection_scores.items()},
            "all_word_counts": {k: v for k, v in word_counts.items()}
        }

    def analyze_dataset_balance(self, data: List[Dict]) -> Dict:
        """
        Analyze dataset balance

        Why analyze:
        - Ensure balance across different data sources
        - Avoid dataset bias
        """
        dataset_counts = Counter()
        dataset_reasons = defaultdict(lambda: Counter())
        dataset_scores = defaultdict(list)

        for item in data:
            dataset_type = item.get("dataset_type", "unknown")
            reason = item.get("selection_reason", "unknown")
            score = item.get("selection_score", 0)

            dataset_counts[dataset_type] += 1
            dataset_reasons[dataset_type][reason] += 1
            dataset_scores[dataset_type].append(score)

        # Convert to plain dict
        dataset_reasons_dict = {
            dataset: dict(reasons)
            for dataset, reasons in dataset_reasons.items()
        }

        return {
            "dataset_counts": dict(dataset_counts),
            "dataset_reasons": dataset_reasons_dict,
            "dataset_scores": {k: v for k, v in dataset_scores.items()}
        }

    def analyze_length_statistics(self, data: List[Dict]) -> Dict:
        """
        Analyze length statistics

        Why analyze:
        - Understand complexity distribution of the test data
        - Provide references for experimental design
        """
        all_word_counts = []
        text_lengths = []  # Character-level length
        dataset_word_counts = defaultdict(list)
        reason_word_counts = defaultdict(list)

        for item in data:
            word_count = item.get("word_count", 0)
            text = item.get("text", "")
            dataset_type = item.get("dataset_type", "unknown")
            reason = item.get("selection_reason", "unknown")

            all_word_counts.append(word_count)
            text_lengths.append(len(text))
            dataset_word_counts[dataset_type].append(word_count)
            reason_word_counts[reason].append(word_count)

        def get_stats(lengths):
            if not lengths:
                return {}
            return {
                "mean": float(np.mean(lengths)),
                "median": float(np.median(lengths)),
                "std": float(np.std(lengths)),
                "min": float(np.min(lengths)),
                "max": float(np.max(lengths)),
                "q25": float(np.percentile(lengths, 25)),
                "q75": float(np.percentile(lengths, 75)),
                "q95": float(np.percentile(lengths, 95))
            }

        return {
            "word_count_stats": get_stats(all_word_counts),
            "text_length_stats": get_stats(text_lengths),
            "dataset_word_stats": {k: get_stats(v) for k, v in dataset_word_counts.items()},
            "reason_word_stats": {k: get_stats(v) for k, v in reason_word_counts.items()},
            "all_word_counts": all_word_counts,
            "all_text_lengths": text_lengths
        }

    def analyze_text_complexity(self, data: List[Dict]) -> Dict:
        """
        Analyze text complexity (multi-dimensional)

        Why analyze:
        - Estimate memorization difficulty for different samples
        - Provide evidence for interpreting results
        """
        complexity_metrics = []
        dataset_complexity = defaultdict(list)
        reason_complexity = defaultdict(list)

        for item in data:
            text = item.get("text", "")
            dataset_type = item.get("dataset_type", "unknown")
            reason = item.get("selection_reason", "unknown")

            if not text:
                continue

            # Compute multi-dimensional complexity metrics
            words = text.split()
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

            metrics = {
                "text_length": len(text),
                "word_count": len(words),
                "unique_words": len(set(words)),
                "word_diversity": len(set(words)) / len(words) if words else 0,
                "unique_chars": len(set(text)),
                "char_diversity": len(set(text)) / len(text) if text else 0,
                "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
                "max_word_length": max([len(w) for w in words]) if words else 0,
                "num_sentences": len(sentences),
                "avg_sentence_length": np.mean([len(s.split()) for s in sentences]) if sentences else 0,
                "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0,
                "digit_ratio": sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
                "punctuation_ratio": sum(1 for c in text if c in '.,;:!?-()[]{}') / len(text) if text else 0,
                "whitespace_ratio": sum(1 for c in text if c.isspace()) / len(text) if text else 0,
            }

            complexity_metrics.append(metrics)

            # Compute an overall complexity score
            complexity_score = (
                metrics["word_diversity"] * 0.3 +
                metrics["char_diversity"] * 0.2 +
                (metrics["avg_word_length"] / 10) * 0.2 +
                (metrics["avg_sentence_length"] / 20) * 0.3
            )
            metrics["complexity_score"] = complexity_score

            dataset_complexity[dataset_type].append(complexity_score)
            reason_complexity[reason].append(complexity_score)

        # Aggregate statistics
        aggregated = {}
        if complexity_metrics:
            for key in complexity_metrics[0].keys():
                values = [m[key] for m in complexity_metrics]
                aggregated[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values))
                }

        return {
            "overall_complexity": aggregated,
            "dataset_complexity": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                                  for k, v in dataset_complexity.items()},
            "reason_complexity": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                                 for k, v in reason_complexity.items()},
            "all_metrics": complexity_metrics
        }

    def analyze_score_distribution(self, data: List[Dict]) -> Dict:
        """
        Analyze selection-score distribution

        Why analyze:
        - Understand the distribution of data quality
        - Identify characteristics of high-quality / low-quality data
        """
        all_scores = []
        dataset_scores = defaultdict(list)
        reason_scores = defaultdict(list)
        score_word_correlation = []

        for item in data:
            score = item.get("selection_score", 0)
            dataset_type = item.get("dataset_type", "unknown")
            reason = item.get("selection_reason", "unknown")
            word_count = item.get("word_count", 0)

            all_scores.append(score)
            dataset_scores[dataset_type].append(score)
            reason_scores[reason].append(score)
            score_word_correlation.append((score, word_count))

        # Compute correlation between score and word count
        if len(score_word_correlation) > 1:
            scores, word_counts = zip(*score_word_correlation)
            correlation = np.corrcoef(scores, word_counts)[0, 1]
        else:
            correlation = 0.0

        def get_score_stats(scores):
            if not scores:
                return {}
            return {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores)),
                "q25": float(np.percentile(scores, 25)),
                "q75": float(np.percentile(scores, 75))
            }

        return {
            "overall_score_stats": get_score_stats(all_scores),
            "dataset_score_stats": {k: get_score_stats(v) for k, v in dataset_scores.items()},
            "reason_score_stats": {k: get_score_stats(v) for k, v in reason_scores.items()},
            "score_word_correlation": float(correlation),
            "all_scores": all_scores,
            "score_bins": np.histogram(all_scores, bins=20)[0].tolist() if all_scores else []
        }

    def analyze_text_features(self, data: List[Dict]) -> Dict:
        """
        In-depth analysis of text features

        Why analyze:
        - Identify linguistic characteristics across different data types
        - Provide evidence for memorization difficulty assessment
        """
        features = {
            "instruction_indicators": defaultdict(int),
            "factual_indicators": defaultdict(int),
            "question_patterns": defaultdict(int),
            "special_chars": defaultdict(int)
        }

        # Instruction-related words
        instruction_words = ["explain", "describe", "write", "create", "generate",
                           "please", "how", "what", "why", "can you", "tell me"]

        # Factual markers
        factual_markers = [r'\b\d{4}\b',  # Year
                          r'\d+%',  # Percentage
                          r'\$\d+',  # Currency
                          r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b']  # Proper nouns

        # Question patterns
        question_patterns = [r'\?$', r'^What ', r'^How ', r'^Why ', r'^When ', r'^Where ']

        for item in data:
            text = item.get("text", "")
            text_lower = text.lower()
            reason = item.get("selection_reason", "unknown")

            # Check instruction words
            for word in instruction_words:
                if word in text_lower:
                    features["instruction_indicators"][reason] += 1
                    break

            # Check factual markers
            for pattern in factual_markers:
                if re.search(pattern, text):
                    features["factual_indicators"][reason] += 1
                    break

            # Check question patterns
            for pattern in question_patterns:
                if re.search(pattern, text):
                    features["question_patterns"][reason] += 1
                    break

            # Count special characters
            features["special_chars"][reason] += sum(1 for c in text if not c.isalnum() and not c.isspace())

        return {
            "instruction_indicators": dict(features["instruction_indicators"]),
            "factual_indicators": dict(features["factual_indicators"]),
            "question_patterns": dict(features["question_patterns"]),
            "special_chars": dict(features["special_chars"])
        }

    def analyze_cross_dataset_comparison(self, data: List[Dict]) -> Dict:
        """
        Cross-dataset comparative analysis

        Why analyze:
        - Identify differences across datasets
        - Provide references for experimental design
        """
        dataset_metrics = defaultdict(lambda: {
            "word_counts": [],
            "scores": [],
            "complexity": [],
            "text_lengths": []
        })

        for item in data:
            dataset_type = item.get("dataset_type", "unknown")
            word_count = item.get("word_count", 0)
            score = item.get("selection_score", 0)
            text = item.get("text", "")

            dataset_metrics[dataset_type]["word_counts"].append(word_count)
            dataset_metrics[dataset_type]["scores"].append(score)
            dataset_metrics[dataset_type]["text_lengths"].append(len(text))

            # Simplified complexity computation
            words = text.split()
            if words:
                complexity = len(set(words)) / len(words)
                dataset_metrics[dataset_type]["complexity"].append(complexity)

        # Compute summary statistics for each dataset
        comparison = {}
        for dataset, metrics in dataset_metrics.items():
            comparison[dataset] = {
                "sample_count": len(metrics["word_counts"]),
                "avg_word_count": float(np.mean(metrics["word_counts"])) if metrics["word_counts"] else 0,
                "avg_score": float(np.mean(metrics["scores"])) if metrics["scores"] else 0,
                "avg_complexity": float(np.mean(metrics["complexity"])) if metrics["complexity"] else 0,
                "avg_text_length": float(np.mean(metrics["text_lengths"])) if metrics["text_lengths"] else 0
            }

        return comparison

    def plot_analysis_results(self, analysis_results: Dict, output_dir: str, dataset_name: str):
        """Plot analysis result figures"""
        os.makedirs(output_dir, exist_ok=True)

        # Create multiple figure files

        # Figure 1: Selection-reason distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1.1 Selection-reason counts
        ax = axes[0, 0]
        reason_counts = analysis_results["selection_distribution"]["selection_reason_counts"]
        if reason_counts:
            ax.bar(range(len(reason_counts)), list(reason_counts.values()))
            ax.set_xticks(range(len(reason_counts)))
            ax.set_xticklabels(list(reason_counts.keys()), rotation=45, ha='right')
            ax.set_title("Selection Reason Distribution")
            ax.set_ylabel("Count")

        # 1.2 Dataset distribution
        ax = axes[0, 1]
        dataset_counts = analysis_results["dataset_balance"]["dataset_counts"]
        if dataset_counts:
            colors = plt.cm.Set3(range(len(dataset_counts)))
            ax.pie(dataset_counts.values(), labels=dataset_counts.keys(),
                  autopct='%1.1f%%', colors=colors)
            ax.set_title("Dataset Distribution")

        # 1.3 Word-count distribution
        ax = axes[1, 0]
        word_counts = analysis_results["length_statistics"]["all_word_counts"]
        if word_counts:
            ax.hist(word_counts, bins=30, edgecolor='black', alpha=0.7)
            ax.set_title("Word Count Distribution")
            ax.set_xlabel("Word Count")
            ax.set_ylabel("Frequency")
            ax.axvline(np.mean(word_counts), color='r', linestyle='--',
                      label=f'Mean: {np.mean(word_counts):.1f}')
            ax.legend()

        # 1.4 Score distribution
        ax = axes[1, 1]
        scores = analysis_results["score_distribution"]["all_scores"]
        if scores:
            ax.hist(scores, bins=30, edgecolor='black', alpha=0.7, color='green')
            ax.set_title("Selection Score Distribution")
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            ax.axvline(np.mean(scores), color='r', linestyle='--',
                      label=f'Mean: {np.mean(scores):.1f}')
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"basic_distribution_{dataset_name}.png"), dpi=300)
        plt.close()

        # Figure 2: Complexity analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        complexity_data = analysis_results["text_complexity"]["all_metrics"]
        if complexity_data:
            # 2.1 Word diversity
            ax = axes[0, 0]
            word_diversity = [m["word_diversity"] for m in complexity_data]
            ax.hist(word_diversity, bins=30, edgecolor='black', alpha=0.7)
            ax.set_title("Word Diversity Distribution")
            ax.set_xlabel("Word Diversity")
            ax.set_ylabel("Frequency")

            # 2.2 Average word length
            ax = axes[0, 1]
            avg_word_length = [m["avg_word_length"] for m in complexity_data]
            ax.hist(avg_word_length, bins=30, edgecolor='black', alpha=0.7, color='orange')
            ax.set_title("Average Word Length Distribution")
            ax.set_xlabel("Average Word Length")
            ax.set_ylabel("Frequency")

            # 2.3 Average sentence length
            ax = axes[1, 0]
            avg_sentence_length = [m["avg_sentence_length"] for m in complexity_data]
            ax.hist(avg_sentence_length, bins=30, edgecolor='black', alpha=0.7, color='purple')
            ax.set_title("Average Sentence Length Distribution")
            ax.set_xlabel("Average Sentence Length (words)")
            ax.set_ylabel("Frequency")

            # 2.4 Overall complexity score
            ax = axes[1, 1]
            complexity_scores = [m["complexity_score"] for m in complexity_data]
            ax.hist(complexity_scores, bins=30, edgecolor='black', alpha=0.7, color='red')
            ax.set_title("Overall Complexity Score Distribution")
            ax.set_xlabel("Complexity Score")
            ax.set_ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"complexity_analysis_{dataset_name}.png"), dpi=300)
        plt.close()

        # Figure 3: Cross-dimension comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 3.1 Boxplot of word counts by reason
        ax = axes[0, 0]
        reason_word_data = analysis_results["selection_distribution"]["all_word_counts"]
        if reason_word_data:
            data_to_plot = [counts for counts in reason_word_data.values() if counts]
            labels = [reason for reason, counts in reason_word_data.items() if counts]
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_title("Word Count by Selection Reason")
                ax.set_ylabel("Word Count")

        # 3.2 Boxplot of scores by reason
        ax = axes[0, 1]
        reason_score_data = analysis_results["selection_distribution"]["all_scores"]
        if reason_score_data:
            data_to_plot = [scores for scores in reason_score_data.values() if scores]
            labels = [reason for reason, scores in reason_score_data.items() if scores]
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_title("Selection Score by Reason")
                ax.set_ylabel("Score")

        # 3.3 Dataset-reason heatmap
        ax = axes[1, 0]
        dataset_reasons = analysis_results["dataset_balance"]["dataset_reasons"]
        if dataset_reasons:
            datasets = list(dataset_reasons.keys())
            all_reasons = set()
            for reasons in dataset_reasons.values():
                all_reasons.update(reasons.keys())
            all_reasons = sorted(list(all_reasons))

            matrix = []
            for dataset in datasets:
                row = []
                for reason in all_reasons:
                    row.append(dataset_reasons[dataset].get(reason, 0))
                matrix.append(row)

            if matrix and all_reasons:
                im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
                ax.set_xticks(range(len(all_reasons)))
                ax.set_yticks(range(len(datasets)))
                ax.set_xticklabels(all_reasons, rotation=45, ha='right')
                ax.set_yticklabels(datasets)
                ax.set_title("Dataset-Reason Heatmap")
                plt.colorbar(im, ax=ax)

        # 3.4 Scatter: score vs word count correlation
        ax = axes[1, 1]
        scores = analysis_results["score_distribution"]["all_scores"]
        word_counts = analysis_results["length_statistics"]["all_word_counts"]
        if scores and word_counts and len(scores) == len(word_counts):
            ax.scatter(word_counts, scores, alpha=0.5, s=10)
            ax.set_title(f"Score vs Word Count\n(correlation: {analysis_results['score_distribution']['score_word_correlation']:.3f})")
            ax.set_xlabel("Word Count")
            ax.set_ylabel("Selection Score")

            # Add trend line
            if len(word_counts) > 1:
                z = np.polyfit(word_counts, scores, 1)
                p = np.poly1d(z)
                ax.plot(sorted(word_counts), p(sorted(word_counts)), "r--", alpha=0.8, linewidth=2)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"cross_dimension_analysis_{dataset_name}.png"), dpi=300)
        plt.close()

    def generate_report(self, analysis_results: Dict, output_path: str, dataset_name: str):
        """Generate a detailed analysis report"""
        report = []
        report.append(f"# Memorization测试数据分析报告 - {dataset_name}\n\n")

        # 1. Basic statistics
        report.append("## 1. 基本统计\n\n")

        # 1.1 Number of samples
        reason_counts = analysis_results["selection_distribution"]["selection_reason_counts"]
        total_samples = sum(reason_counts.values())
        report.append(f"- **总样本数**: {total_samples}\n\n")

        # 1.2 Selection reason distribution
        report.append("### 1.1 选择原因分布\n\n")
        for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            report.append(f"- `{reason}`: {count} ({percentage:.1f}%)\n")
        report.append("\n")

        # 1.3 Dataset distribution
        report.append("### 1.2 数据集分布\n\n")
        dataset_counts = analysis_results["dataset_balance"]["dataset_counts"]
        for dataset, count in sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            report.append(f"- `{dataset}`: {count} ({percentage:.1f}%)\n")
        report.append("\n")

        # 2. Length statistics
        report.append("## 2. 长度统计\n\n")

        word_stats = analysis_results["length_statistics"]["word_count_stats"]
        if word_stats:
            report.append("### 2.1 词数统计\n\n")
            report.append(f"- 平均词数: {word_stats['mean']:.1f}\n")
            report.append(f"- 中位数: {word_stats['median']:.1f}\n")
            report.append(f"- 标准差: {word_stats['std']:.1f}\n")
            report.append(f"- 范围: [{word_stats['min']:.0f}, {word_stats['max']:.0f}]\n")
            report.append(f"- 四分位数: Q25={word_stats['q25']:.1f}, Q75={word_stats['q75']:.1f}\n")
            report.append(f"- 95百分位: {word_stats['q95']:.1f}\n\n")

        # 2.2 Length by dataset
        report.append("### 2.2 各数据集词数统计\n\n")
        dataset_word_stats = analysis_results["length_statistics"]["dataset_word_stats"]
        for dataset, stats in dataset_word_stats.items():
            if stats:
                report.append(f"**{dataset}**:\n")
                report.append(f"  - 平均: {stats['mean']:.1f} ± {stats['std']:.1f}\n")
                report.append(f"  - 范围: [{stats['min']:.0f}, {stats['max']:.0f}]\n\n")

        # 3. Score analysis
        report.append("## 3. 选择分数分析\n\n")

        score_stats = analysis_results["score_distribution"]["overall_score_stats"]
        if score_stats:
            report.append("### 3.1 总体分数统计\n\n")
            report.append(f"- 平均分数: {score_stats['mean']:.2f}\n")
            report.append(f"- 中位数: {score_stats['median']:.2f}\n")
            report.append(f"- 标准差: {score_stats['std']:.2f}\n")
            report.append(f"- 范围: [{score_stats['min']:.2f}, {score_stats['max']:.2f}]\n\n")

        # 3.2 Scores by selection reason
        report.append("### 3.2 各选择原因的分数统计\n\n")
        reason_score_stats = analysis_results["score_distribution"]["reason_score_stats"]
        for reason, stats in sorted(reason_score_stats.items(), key=lambda x: x[1].get('mean', 0), reverse=True):
            if stats:
                report.append(f"**{reason}**:\n")
                report.append(f"  - 平均: {stats['mean']:.2f} ± {stats['std']:.2f}\n")
                report.append(f"  - 范围: [{stats['min']:.2f}, {stats['max']:.2f}]\n\n")

        # 3.3 Correlation between score and word count
        correlation = analysis_results["score_distribution"]["score_word_correlation"]
        report.append("### 3.3 分数与词数相关性\n\n")
        report.append(f"- Pearson相关系数: {correlation:.3f}\n")
        if abs(correlation) < 0.3:
            report.append("  - 解释: 弱相关或无相关\n\n")
        elif abs(correlation) < 0.7:
            report.append("  - 解释: 中等相关\n\n")
        else:
            report.append("  - 解释: 强相关\n\n")

        # 4. Text complexity analysis
        report.append("## 4. 文本复杂度分析\n\n")

        complexity = analysis_results["text_complexity"]["overall_complexity"]
        if complexity:
            report.append("### 4.1 总体复杂度指标\n\n")

            key_metrics = [
                ("word_diversity", "词汇多样性"),
                ("char_diversity", "字符多样性"),
                ("avg_word_length", "平均词长"),
                ("avg_sentence_length", "平均句长"),
                ("complexity_score", "综合复杂度分数")
            ]

            for key, name in key_metrics:
                if key in complexity:
                    stats = complexity[key]
                    report.append(f"**{name}**:\n")
                    report.append(f"  - 平均: {stats['mean']:.3f} ± {stats['std']:.3f}\n")
                    report.append(f"  - 范围: [{stats['min']:.3f}, {stats['max']:.3f}]\n\n")

        # 4.2 Complexity by dataset
        report.append("### 4.2 各数据集复杂度比较\n\n")
        dataset_complexity = analysis_results["text_complexity"]["dataset_complexity"]
        for dataset, stats in sorted(dataset_complexity.items(), key=lambda x: x[1]['mean'], reverse=True):
            report.append(f"**{dataset}**:\n")
            report.append(f"  - 平均复杂度: {stats['mean']:.3f} ± {stats['std']:.3f}\n\n")

        # 5. Text feature analysis
        report.append("## 5. 文本特征分析\n\n")

        text_features = analysis_results["text_features"]

        if text_features["instruction_indicators"]:
            report.append("### 5.1 指令特征分布\n\n")
            for reason, count in sorted(text_features["instruction_indicators"].items(),
                                       key=lambda x: x[1], reverse=True):
                report.append(f"- `{reason}`: {count} 个样本包含指令词汇\n")
            report.append("\n")

        if text_features["factual_indicators"]:
            report.append("### 5.2 事实性特征分布\n\n")
            for reason, count in sorted(text_features["factual_indicators"].items(),
                                       key=lambda x: x[1], reverse=True):
                report.append(f"- `{reason}`: {count} 个样本包含事实性标记\n")
            report.append("\n")

        # 6. Cross-dataset comparison
        report.append("## 6. 跨数据集比较\n\n")

        cross_comparison = analysis_results["cross_comparison"]
        if cross_comparison:
            # Create comparison table
            report.append("| 数据集 | 样本数 | 平均词数 | 平均分数 | 平均复杂度 | 平均文本长度 |\n")
            report.append("|--------|--------|----------|----------|------------|-------------|\n")

            for dataset, metrics in sorted(cross_comparison.items(),
                                          key=lambda x: x[1]['sample_count'], reverse=True):
                report.append(f"| {dataset} | {metrics['sample_count']} | "
                            f"{metrics['avg_word_count']:.1f} | "
                            f"{metrics['avg_score']:.2f} | "
                            f"{metrics['avg_complexity']:.3f} | "
                            f"{metrics['avg_text_length']:.1f} |\n")
            report.append("\n")

        # 7. Data quality assessment
        report.append("## 7. 数据质量评估\n\n")

        # 7.1 Coverage assessment
        report.append("### 7.1 覆盖度评估\n\n")
        num_datasets = len(dataset_counts)
        num_reasons = len(reason_counts)
        report.append(f"- 涵盖数据集数量: {num_datasets}\n")
        report.append(f"- 选择原因类型数: {num_reasons}\n")
        report.append(f"- 总样本数: {total_samples}\n\n")

        # 7.2 Balance assessment
        report.append("### 7.2 平衡性评估\n\n")
        reason_values = list(reason_counts.values())
        if reason_values:
            max_count = max(reason_values)
            min_count = min(reason_values)
            balance_ratio = min_count / max_count if max_count > 0 else 0
            report.append(f"- 最大类别样本数: {max_count}\n")
            report.append(f"- 最小类别样本数: {min_count}\n")
            report.append(f"- 平衡比率: {balance_ratio:.2f}\n")

            if balance_ratio > 0.5:
                report.append("  - 评价: 数据集平衡性较好\n\n")
            elif balance_ratio > 0.2:
                report.append("  - 评价: 数据集平衡性一般\n\n")
            else:
                report.append("  - 评价: 数据集存在明显不平衡\n\n")

        # 8. Recommendations and conclusions
        report.append("## 8. 建议和结论\n\n")

        report.append("### 8.1 数据特点总结\n\n")

        # Provide recommendations based on results
        if word_stats:
            if word_stats['std'] / word_stats['mean'] > 0.5:
                report.append("- 词数分布较为分散，样本长度差异较大\n")
            else:
                report.append("- 词数分布较为集中，样本长度相对一致\n")

        if complexity:
            avg_complexity = complexity.get("complexity_score", {}).get("mean", 0)
            if avg_complexity > 0.5:
                report.append("- 文本整体复杂度较高，可能对memorization造成一定挑战\n")
            else:
                report.append("- 文本整体复杂度适中，适合作为测试数据\n")

        report.append("\n### 8.2 实验建议\n\n")
        report.append("- 建议按照选择原因分组进行对比实验\n")
        report.append("- 建议考虑文本长度作为控制变量\n")
        report.append("- 建议关注复杂度与memorization效果的关系\n")

        # Save report
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(report)


def main():
    parser = argparse.ArgumentParser(description="分析测试数据")
    parser.add_argument("--input_dir", type=str, default="../../data/pretraining_test_data/mem_test", help="输入目录")
    parser.add_argument("--output_dir", type=str, default="../../data/pretraining_test_data/analysis", help="输出目录")
    parser.add_argument("--datasets", nargs="+", default=["wiki"],  help="要处理的数据集列表")
    args = parser.parse_args()

    analyzer = TestDataAnalyzer()
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each dataset
    for dataset_name in args.datasets:
        # Find the file according to the new naming convention
        suffix_map = {
            "stackexchange": "instruction",
            "wiki": "fact",
            "dclm": "privacy"
        }

        suffix = suffix_map.get(dataset_name, "")
        test_path = os.path.join(args.input_dir, f"{dataset_name}_{suffix}.jsonl")
        if not test_path:
            print(f"警告: 找不到 {dataset_name} 的数据文件，跳过")
            continue

        print(f"\n{'='*60}")
        print(f"分析数据集: {dataset_name}")
        print(f"数据文件: {test_path}")
        print(f"{'='*60}\n")

        data = analyzer.load_data(test_path)
        print(f"加载了 {len(data)} 个样本")

        # Run analyses
        print("执行分析...")
        analysis_results = {
            "selection_distribution": analyzer.analyze_selection_distribution(data),
            "dataset_balance": analyzer.analyze_dataset_balance(data),
            "length_statistics": analyzer.analyze_length_statistics(data),
            "text_complexity": analyzer.analyze_text_complexity(data),
            "score_distribution": analyzer.analyze_score_distribution(data),
            "text_features": analyzer.analyze_text_features(data),
            "cross_comparison": analyzer.analyze_cross_dataset_comparison(data)
        }

        # Print summary
        print(f"\n分析摘要:")
        print(f"- 总样本数: {len(data)}")
        print(f"- 选择原因类型: {list(analysis_results['selection_distribution']['selection_reason_counts'].keys())}")
        print(f"- 数据集来源: {list(analysis_results['dataset_balance']['dataset_counts'].keys())}")

        word_stats = analysis_results['length_statistics']['word_count_stats']
        if word_stats:
            print(f"- 平均词数: {word_stats['mean']:.1f} ± {word_stats['std']:.1f}")

        score_stats = analysis_results['score_distribution']['overall_score_stats']
        if score_stats:
            print(f"- 平均分数: {score_stats['mean']:.2f} ± {score_stats['std']:.2f}")

        # Generate visualizations
        print("\n生成可视化图表...")
        analyzer.plot_analysis_results(analysis_results, args.output_dir, dataset_name)

        # Generate report
        print("生成分析报告...")
        report_path = os.path.join(args.output_dir, f"analysis_report_{dataset_name}.md")
        analyzer.generate_report(analysis_results, report_path, dataset_name)

        # Save detailed analysis results
        json_path = os.path.join(args.output_dir, f"analysis_results_{dataset_name}.json")

        # Clean non-serializable data (remove large lists, keep only summary stats)
        clean_results = {}
        for key, value in analysis_results.items():
            if isinstance(value, dict):
                clean_value = {}
                for k, v in value.items():
                    # Skip large list-like data
                    if k in ['all_metrics', 'all_word_counts', 'all_text_lengths',
                            'all_scores', 'score_bins']:
                        continue
                    clean_value[k] = v
                clean_results[key] = clean_value

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n分析完成！")
        print(f"- 基础分布图表: {os.path.join(args.output_dir, f'basic_distribution_{dataset_name}.png')}")
        print(f"- 复杂度分析图表: {os.path.join(args.output_dir, f'complexity_analysis_{dataset_name}.png')}")
        print(f"- 跨维度比较图表: {os.path.join(args.output_dir, f'cross_dimension_analysis_{dataset_name}.png')}")
        print(f"- 分析报告: {report_path}")
        print(f"- 详细结果: {json_path}")


if __name__ == "__main__":
    main()
