"""
Instruction Token Probability Analysis Module
Analyzes the probability distribution features of instruction-related tokens
when base and SFT models generate text.

Data Format Description:
- Input file format: {data_type}_prefix{prefix_length}_{model_scale}_{model_type}_{num_samples}_samples.jsonl
- Each JSON line contains: prefix_text, generated_text, generated_tokens, top_tokens, etc. fields.
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd
from scipy.stats import entropy
from WORDS import WORDSSet # Assuming WORDS.py contains the word sets

# Set plot style
sns.set_style("whitegrid")


class TokenClassifier:
    """
    Token Type Classifier
    Used to determine the type of a token (instruction word, question word, politeness word, structural word, etc.).
    """

    def __init__(self):
        # Instruction verbs (including user-provided list)
        self.instruct_verbs = WORDSSet.instruction_verbs

        # Question words
        self.question_words = WORDSSet.question_words

        # Politeness words
        self.polite_words = WORDSSet.politeness_markers

        # Structural marker words
        self.structure_words = WORDSSet.structure_markers

        # Modal/auxiliary words
        self.modal_words = WORDSSet.modal_words

    def classify_token(self, token_text: str) -> Set[str]:
        """
        Classifies a single token.
        Input: token text string
        Output: A set of types the token belongs to (a token can belong to multiple types).
        """
        # Clean the token text: remove leading/trailing spaces, convert to lowercase
        token_clean = token_text.strip().lower()
        categories = set()

        # Check if it contains an instruction word
        if any(verb in token_clean for verb in self.instruct_verbs):
            categories.add("Instruction")

        # Check if it contains a question word
        if any(q in token_clean for q in self.question_words):
            categories.add("Question")

        # Check if it contains a politeness word
        if any(p in token_clean for p in self.polite_words):
            categories.add("Politeness")

        # Check if it contains a structural word
        if any(s in token_clean for s in self.structure_words):
            categories.add("Structural")

        # Check if it contains a modal word
        if any(m in token_clean for m in self.modal_words):
            categories.add("Modal")

        # If no type is matched, classify as "Other"
        if not categories:
            categories.add("Other")

        return categories


class TokenProbabilityAnalyzer:
    """
    Token Probability Analyzer
    Calculates statistical metrics like entropy, Top-K ratio, and rank of token probabilities.
    """

    def __init__(self, classifier: TokenClassifier, top_k: int = 10):
        """
        Initializes the analyzer.

        Args:
            classifier: An instance of the TokenClassifier.
            top_k: The number of top probability tokens to analyze, default is 10.
        """
        self.classifier = classifier
        self.top_k = top_k

    def analyze_generation(self, generation_data: Dict) -> Dict:
        """
        Analyzes the generated content of a single sample.

        Input:
            generation_data: A dictionary containing generation information, including:
                - generated_tokens: A list of generated token IDs.
                - top_tokens: A list of top-K token information for each position.

        Output:
            A dictionary containing various statistical metrics:
                - token_type_entropy: A list of entropy values for each token type.
                - token_type_prob: A list of probabilities for each token type.
                - top_k_type_counts: A count of each token type within the Top-K.
                - first_instruct_rank: The position and rank of the first instruction word.
                - token_details: Detailed analysis information for each token.
        """
        results = {
            "token_type_entropy": defaultdict(list),
            "token_type_prob": defaultdict(list),
            "top_k_type_counts": defaultdict(int),
            "first_instruct_rank": None,
            "token_details": []
        }

        generated_tokens = generation_data.get("generated_tokens", [])
        top_tokens_list = generation_data.get("top_tokens", [])

        # Ensure data lengths are consistent
        if len(generated_tokens) != len(top_tokens_list):
            print(f"Warning: length of generated_tokens ({len(generated_tokens)}) != length of top_tokens ({len(top_tokens_list)})")
            min_len = min(len(generated_tokens), len(top_tokens_list))
            generated_tokens = generated_tokens[:min_len]
            top_tokens_list = top_tokens_list[:min_len]

        # Iterate through each generated position
        for pos_idx, (token_id, top_tokens_info) in enumerate(zip(generated_tokens, top_tokens_list)):
            if not top_tokens_info:
                continue

            # 1. Get information about the actually generated token
            generated_token_info = None
            for token_info in top_tokens_info:
                if token_info.get("token_id") == token_id:
                    generated_token_info = token_info
                    break

            if not generated_token_info:
                # If the generated token is not in top-K, use the first one as an approximation
                generated_token_info = top_tokens_info[0]

            token_text = generated_token_info.get("token_text", "")
            token_prob = generated_token_info.get("probability", 0.0)
            token_rank = generated_token_info.get("rank", -1)

            # 2. Classify the currently generated token
            token_types = self.classifier.classify_token(token_text)

            # 3. Calculate the entropy of the probability distribution at the current position
            all_probs = [t.get("probability", 0.0) for t in top_tokens_info]
            if all_probs:
                # Normalize probabilities (in case the sum is not 1)
                prob_sum = sum(all_probs)
                if prob_sum > 0:
                    normalized_probs = [p / prob_sum for p in all_probs]
                else:
                    normalized_probs = all_probs

                # Calculate entropy (measure of uncertainty)
                pos_entropy = entropy(normalized_probs + [1e-10] * len(normalized_probs))
            else:
                pos_entropy = 0.0

            # 4. Count the number of each token type in the Top-K
            top_k_type_distribution = defaultdict(int)
            for top_token_info in top_tokens_info[:self.top_k]:
                top_token_text = top_token_info.get("token_text", "")
                top_token_types = self.classifier.classify_token(top_token_text)
                for t_type in top_token_types:
                    top_k_type_distribution[t_type] += 1
                    results["top_k_type_counts"][t_type] += 1

            # 5. Record the rank of the first instruction word
            if results["first_instruct_rank"] is None and "Instruction" in token_types:
                results["first_instruct_rank"] = {
                    "position": pos_idx,
                    "rank": token_rank,
                    "token_text": token_text,
                    "probability": token_prob
                }

            # 6. Record detailed information for the current token
            token_detail = {
                "position": pos_idx,
                "token_id": token_id,
                "token_text": token_text,
                "types": list(token_types),
                "probability": float(token_prob),
                "rank": int(token_rank),
                "entropy": float(pos_entropy),
                "top_k_type_dist": dict(top_k_type_distribution)
            }
            results["token_details"].append(token_detail)

            # 7. Update statistics for each type
            for t_type in token_types:
                results["token_type_entropy"][t_type].append(pos_entropy)
                results["token_type_prob"][t_type].append(token_prob)

        return results

    def compute_statistics(self, results: Dict) -> Dict:
        """
        Computes summary statistics.

        Input:
            results: The output of analyze_generation.

        Output:
            A dictionary of summary statistics.
        """
        stats = {
            "sample_summary": {},
            "token_type_stats": {},
            "top_k_dist_stats": {},
            "first_instruction_word": results.get("first_instruct_rank", None)
        }

        total_tokens = len(results.get("token_details", []))
        stats["sample_summary"]["total_tokens"] = total_tokens

        # 1. Calculate average entropy and probability for each token type
        for token_type, entropies in results["token_type_entropy"].items():
            if entropies:
                stats["token_type_stats"][token_type] = {
                    "avg_entropy": float(np.mean(entropies)),
                    "std_entropy": float(np.std(entropies)),
                    "avg_probability": float(np.mean(results["token_type_prob"][token_type])),
                    "std_probability": float(np.std(results["token_type_prob"][token_type])),
                    "token_count": len(entropies),
                    "token_ratio": len(entropies) / total_tokens if total_tokens > 0 else 0
                }

        # 2. Calculate the ratio of each type in the Top-K
        total_top_k_counts = sum(results["top_k_type_counts"].values())
        for token_type, count in results["top_k_type_counts"].items():
            stats["top_k_dist_stats"][token_type] = {
                "count_in_top_k": count,
                "ratio_in_top_k": count / total_top_k_counts if total_top_k_counts > 0 else 0
            }

        return stats


class InstructionTokenAnalyzer:
    """
    Main analysis class.
    Integrates all analysis workflows, handles multiple models and datasets.
    """

    def __init__(self, args):
        """
        Initializes the analyzer.

        Args:
            args: Command-line arguments object.
        """
        self.args = args
        self.classifier = TokenClassifier()
        self.analyzer = TokenProbabilityAnalyzer(self.classifier, top_k=args.top_k)

        # Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.vis_dir, exist_ok=True)

        print(f"\nInitialization complete.")
        print(f"  - Output directory: {args.output_dir}")
        print(f"  - Visualization directory: {args.vis_dir}")

    def load_model_outputs(self, file_path: str) -> List[Dict]:
        """
        Loads the generated content from a model.

        Input:
            file_path: The full path to the model output file.

        Output:
            A list of generated content, where each element is a dictionary.
        """
        outputs = []

        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return outputs

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    outputs.append(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON parsing failed on line {line_num} - {e}")
                    continue

        print(f"  Successfully loaded {len(outputs)} samples.")
        return outputs

    def process_single_model(self,
                            model_scale: str,
                            model_type: str,
                            data_type: str,
                            prefix_length: int = 16,
                            num_samples: int = 50) -> Dict:
        """
        Processes the generated content from a single model on a specific dataset.

        Input:
            model_scale: The model scale (e.g., "1B", "7B", "13B", "32B").
            model_type: The model type ("base" or "sft").
            data_type: The data type (e.g., "stackexchange", "wiki-fact", "dclm-privacy").
            prefix_length: The prefix length, default 16.
            num_samples: The number of samples, default 50.

        Output:
            A dictionary of analysis results.
        """
        # Build the input file path: {data_type}_prefix{prefix_length}_{model_scale}_{model_type}_{num_samples}_samples.jsonl
        filename = f"{data_type}_prefix{prefix_length}_{model_scale}_{model_type}_{num_samples}_samples.jsonl"
        input_file = os.path.join(self.args.input_dir, filename)

        print(f"\nProcessing: {model_scale} {model_type} - {data_type}")
        print(f"  File: {filename}")

        # Load model outputs
        model_outputs = self.load_model_outputs(input_file)
        if not model_outputs:
            return None

        # Analyze each sample
        all_sample_results = []
        all_sample_stats = []

        for idx, output in enumerate(model_outputs):
            # Analyze a single sample
            result = self.analyzer.analyze_generation(output)

            # Compute statistics for this sample
            stats = self.analyzer.compute_statistics(result)

            # Add sample identification info
            result["sample_id"] = output.get("sample_id", idx)
            result["prefix_text"] = output.get("prefix_text", "")
            result["generated_text"] = output.get("generated_text", "")

            all_sample_results.append(result)
            all_sample_stats.append(stats)

        # Aggregate statistics from all samples
        combined_stats = self._aggregate_statistics(all_sample_stats)

        # Save detailed results
        output_data = {
            "meta_info": {
                "model_scale": model_scale,
                "model_type": model_type,
                "data_type": data_type,
                "prefix_length": prefix_length,
                "num_samples": len(all_sample_results),
                "top_k": self.args.top_k
            },
            "aggregated_statistics": combined_stats,
            "sample_results": all_sample_results
        }

        # Save to a file
        output_filename = f"analysis_{data_type}_{model_scale}_{model_type}.json"
        output_file = os.path.join(self.args.output_dir, output_filename)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"  ✓ Results saved to: {output_filename}")

        return combined_stats

    def _aggregate_statistics(self, stats_list: List[Dict]) -> Dict:
        """
        Aggregates statistics from multiple samples.

        Input:
            stats_list: A list of statistics dictionaries from multiple samples.

        Output:
            Aggregated statistics.
        """
        aggregated = {
            "num_samples": len(stats_list),
            "agg_token_type_stats": defaultdict(lambda: {
                "entropy_list": [],
                "probability_list": [],
                "token_count_list": [],
                "token_ratio_list": []
            }),
            "agg_top_k_dist_stats": defaultdict(lambda: {
                "count_list": [],
                "ratio_list": []
            }),
            "first_instruct_stats": {
                "occurrence_count": 0,
                "positions": [],
                "ranks": [],
                "probabilities": []
            }
        }

        # Collect data from all samples
        for stats in stats_list:
            # Token type stats
            for token_type, type_stats in stats.get("token_type_stats", {}).items():
                aggregated["agg_token_type_stats"][token_type]["entropy_list"].append(type_stats["avg_entropy"])
                aggregated["agg_token_type_stats"][token_type]["probability_list"].append(type_stats["avg_probability"])
                aggregated["agg_token_type_stats"][token_type]["token_count_list"].append(type_stats["token_count"])
                aggregated["agg_token_type_stats"][token_type]["token_ratio_list"].append(type_stats["token_ratio"])

            # Top-K distribution stats
            for token_type, topk_stats in stats.get("top_k_dist_stats", {}).items():
                aggregated["agg_top_k_dist_stats"][token_type]["count_list"].append(topk_stats["count_in_top_k"])
                aggregated["agg_top_k_dist_stats"][token_type]["ratio_list"].append(topk_stats["ratio_in_top_k"])

            # First instruction word stats
            first_instruct = stats.get("first_instruction_word", None)
            if first_instruct:
                aggregated["first_instruct_stats"]["occurrence_count"] += 1
                aggregated["first_instruct_stats"]["positions"].append(first_instruct["position"])
                aggregated["first_instruct_stats"]["ranks"].append(first_instruct["rank"])
                aggregated["first_instruct_stats"]["probabilities"].append(first_instruct["probability"])

        # Calculate summary statistics (mean, std dev, etc.)
        final_stats = {
            "num_samples": aggregated["num_samples"],
            "token_type_summary": {},
            "top_k_dist_summary": {},
            "first_instruct_summary": {}
        }

        # Token type summary
        for token_type, data in aggregated["agg_token_type_stats"].items():
            final_stats["token_type_summary"][token_type] = {
                "avg_entropy": float(np.mean(data["entropy_list"])) if data["entropy_list"] else 0.0,
                "std_entropy": float(np.std(data["entropy_list"])) if data["entropy_list"] else 0.0,
                "avg_probability": float(np.mean(data["probability_list"])) if data["probability_list"] else 0.0,
                "std_probability": float(np.std(data["probability_list"])) if data["probability_list"] else 0.0,
                "avg_token_count": float(np.mean(data["token_count_list"])) if data["token_count_list"] else 0.0,
                "avg_token_ratio": float(np.mean(data["token_ratio_list"])) if data["token_ratio_list"] else 0.0
            }

        # Top-K distribution summary
        for token_type, data in aggregated["agg_top_k_dist_stats"].items():
            final_stats["top_k_dist_summary"][token_type] = {
                "avg_count_in_top_k": float(np.mean(data["count_list"])) if data["count_list"] else 0.0,
                "avg_ratio_in_top_k": float(np.mean(data["ratio_list"])) if data["ratio_list"] else 0.0
            }

        # First instruction word summary
        first_data = aggregated["first_instruct_stats"]
        final_stats["first_instruct_summary"] = {
            "occurrence_count": first_data["occurrence_count"],
            "occurrence_rate": first_data["occurrence_count"] / aggregated["num_samples"] if aggregated["num_samples"] > 0 else 0.0,
            "avg_position": float(np.mean(first_data["positions"])) if first_data["positions"] else None,
            "avg_rank": float(np.mean(first_data["ranks"])) if first_data["ranks"] else None,
            "avg_probability": float(np.mean(first_data["probabilities"])) if first_data["probabilities"] else None
        }

        return final_stats

    def compare_base_sft(self):
        """
        Compares the results of base and SFT models.
        Iterates through all model scales and data types for analysis.
        """
        comparison_results = {}

        for model_scale in self.args.model_scales:
            comparison_results[model_scale] = {}

            for data_type in self.args.data_types:
                print(f"\n{'='*70}")
                print(f"Analyzing {model_scale} model on {data_type} data")
                print(f"{'='*70}")

                # Process base model
                base_stats = self.process_single_model(
                    model_scale, "base", data_type,
                    self.args.prefix_length, self.args.num_samples
                )

                # Process SFT model
                sft_stats = self.process_single_model(
                    model_scale, "sft", data_type,
                    self.args.prefix_length, self.args.num_samples
                )

                # Save comparison results
                if base_stats and sft_stats:
                    comparison_results[model_scale][data_type] = {
                        "base": base_stats,
                        "sft": sft_stats
                    }

        # Save the complete comparison results
        comparison_file = os.path.join(self.args.output_dir, "complete_comparison.json")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*70}")
        print(f"✓ Complete comparison results saved to: {comparison_file}")
        print(f"{'='*70}")

        # Generate visualizations
        self.visualize_comparison(comparison_results)

        # Print summary
        self.print_summary(comparison_results)

        return comparison_results

    def visualize_comparison(self, comparison_results: Dict):
        """
        Visualizes the comparison results.

        Input:
            comparison_results: The dictionary of comparison results.
        """
        print(f"\n{'='*70}")
        print("Generating visualization charts...")
        print(f"{'='*70}\n")

        # 1. Entropy comparison plot
        self._plot_entropy_comparison(comparison_results)

        # 2. Probability comparison plot
        self._plot_probability_comparison(comparison_results)

        # 3. Token type ratio comparison plot
        self._plot_type_ratio_comparison(comparison_results)

        # 4. Top-K distribution comparison plot
        self._plot_topk_distribution_comparison(comparison_results)

        # 5. First instruction word rank comparison plot
        self._plot_first_instruct_rank_comparison(comparison_results)

        print(f"\n✓ All visualization charts have been generated.")

    def _plot_entropy_comparison(self, results: Dict):
        """
        Plots the entropy comparison.
        Compares the average entropy of base and SFT models for different token types.
        """
        token_types = ["Instruction", "Question", "Politeness", "Structural", "Modal", "Other"]

        for data_type in self.args.data_types:
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            fig.suptitle(f'{data_type} Dataset - Token Type Entropy Comparison (Base vs SFT)\n'
                         f'Higher entropy indicates a more uniform/uncertain probability distribution.',
                         fontsize=16, fontweight='bold')

            for idx, model_scale in enumerate(self.args.model_scales):
                ax = axes[idx // 2, idx % 2]

                if model_scale not in results or data_type not in results[model_scale]:
                    ax.text(0.5, 0.5, 'Data Missing', ha='center', va='center', fontsize=14)
                    ax.set_title(f'{model_scale} Model', fontsize=14)
                    continue

                base_data = results[model_scale][data_type]["base"]["token_type_summary"]
                sft_data = results[model_scale][data_type]["sft"]["token_type_summary"]

                base_entropies = [base_data.get(t, {}).get("avg_entropy", 0) for t in token_types]
                sft_entropies = [sft_data.get(t, {}).get("avg_entropy", 0) for t in token_types]

                # Calculate percentage change
                entropy_changes = []
                for b, s in zip(base_entropies, sft_entropies):
                    change = ((s - b) / b) * 100 if b > 0 else 0
                    entropy_changes.append(change)

                x = np.arange(len(token_types))
                width = 0.35

                bars1 = ax.bar(x - width/2, base_entropies, width, label='Base Model',
                               alpha=0.8, color='skyblue', edgecolor='black', linewidth=0.5)
                bars2 = ax.bar(x + width/2, sft_entropies, width, label='SFT Model',
                               alpha=0.8, color='coral', edgecolor='black', linewidth=0.5)

                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.annotate(f'{height:.2f}',
                                        xy=(bar.get_x() + bar.get_width() / 2, height),
                                        xytext=(0, 3),  # 3 points vertical offset
                                        textcoords="offset points",
                                        ha='center', va='bottom', fontsize=9)

                ax.set_xlabel('Token Type', fontsize=12, fontweight='bold')
                ax.set_ylabel('Average Entropy', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_scale} Model', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(token_types, rotation=45, ha='right')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3, axis='y')

                # Add percentage change text
                for i, (b_ent, s_ent, change) in enumerate(zip(base_entropies, sft_entropies, entropy_changes)):
                    if b_ent > 0 or s_ent > 0:
                        y_pos = max(b_ent, s_ent) * 1.1
                        color = 'green' if change < 0 else 'red'
                        ax.text(i, y_pos, f'{change:+.1f}%',
                                ha='center', fontsize=8, color=color, fontweight='bold')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = os.path.join(self.args.vis_dir, f'entropy_comparison_{data_type}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Entropy comparison plot saved: {save_path}")

    def _plot_probability_comparison(self, results: Dict):
        """
        Plots the probability comparison.
        Compares the average probability of base and SFT models for different token types.
        """
        token_types = ["Instruction", "Question", "Politeness", "Structural", "Modal"]

        for data_type in self.args.data_types:
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            fig.suptitle(f'{data_type} Dataset - Token Type Probability Comparison (Base vs SFT)\n'
                         f'Higher probability indicates the model is more confident in that token type.',
                         fontsize=16, fontweight='bold')

            for idx, model_scale in enumerate(self.args.model_scales):
                ax = axes[idx // 2, idx % 2]

                if model_scale not in results or data_type not in results[model_scale]:
                    ax.text(0.5, 0.5, 'Data Missing', ha='center', va='center', fontsize=14)
                    ax.set_title(f'{model_scale} Model', fontsize=14)
                    continue

                base_data = results[model_scale][data_type]["base"]["token_type_summary"]
                sft_data = results[model_scale][data_type]["sft"]["token_type_summary"]

                base_probs = [base_data.get(t, {}).get("avg_probability", 0) for t in token_types]
                sft_probs = [sft_data.get(t, {}).get("avg_probability", 0) for t in token_types]

                x = np.arange(len(token_types))
                width = 0.35

                bars1 = ax.bar(x - width/2, base_probs, width, label='Base Model',
                               alpha=0.8, color='lightgreen', edgecolor='black', linewidth=0.5)
                bars2 = ax.bar(x + width/2, sft_probs, width, label='SFT Model',
                               alpha=0.8, color='orange', edgecolor='black', linewidth=0.5)

                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.annotate(f'{height:.3f}',
                                        xy=(bar.get_x() + bar.get_width() / 2, height),
                                        xytext=(0, 3), textcoords="offset points",
                                        ha='center', va='bottom', fontsize=9)

                ax.set_xlabel('Token Type', fontsize=12, fontweight='bold')
                ax.set_ylabel('Average Probability', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_scale} Model', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(token_types, rotation=45, ha='right')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = os.path.join(self.args.vis_dir, f'probability_comparison_{data_type}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Probability comparison plot saved: {save_path}")

    def _plot_type_ratio_comparison(self, results: Dict):
        """
        Plots the token type ratio comparison.
        Uses a stacked bar chart to show the proportion of different token types.
        """
        token_types = ["Instruction", "Question", "Politeness", "Structural", "Modal", "Other"]

        for data_type in self.args.data_types:
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            fig.suptitle(f'{data_type} Dataset - Token Type Ratio Comparison (Base vs SFT)',
                         fontsize=16, fontweight='bold')

            colors = plt.cm.Set3(np.linspace(0, 1, len(token_types)))

            for idx, model_scale in enumerate(self.args.model_scales):
                ax = axes[idx // 2, idx % 2]

                if model_scale not in results or data_type not in results[model_scale]:
                    ax.text(0.5, 0.5, 'Data Missing', ha='center', va='center', fontsize=14)
                    ax.set_title(f'{model_scale} Model', fontsize=14)
                    continue

                base_data = results[model_scale][data_type]["base"]["token_type_summary"]
                sft_data = results[model_scale][data_type]["sft"]["token_type_summary"]

                base_ratios = [base_data.get(t, {}).get("avg_token_ratio", 0) * 100 for t in token_types]
                sft_ratios = [sft_data.get(t, {}).get("avg_token_ratio", 0) * 100 for t in token_types]

                width = 0.6
                bottom_base = 0
                bottom_sft = 0

                for i, (t_type, color) in enumerate(zip(token_types, colors)):
                    ax.bar([0], [base_ratios[i]], width, bottom=bottom_base,
                           label=t_type if idx == 0 else "", color=color, alpha=0.8,
                           edgecolor='white', linewidth=2)
                    ax.bar([1], [sft_ratios[i]], width, bottom=bottom_sft,
                           color=color, alpha=0.8, edgecolor='white', linewidth=2)

                    if base_ratios[i] > 5:
                        ax.text(0, bottom_base + base_ratios[i]/2,
                                f'{base_ratios[i]:.1f}%',
                                ha='center', va='center', fontsize=10, fontweight='bold')
                    if sft_ratios[i] > 5:
                        ax.text(1, bottom_sft + sft_ratios[i]/2,
                                f'{sft_ratios[i]:.1f}%',
                                ha='center', va='center', fontsize=10, fontweight='bold')

                    bottom_base += base_ratios[i]
                    bottom_sft += sft_ratios[i]

                ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_scale} Model', fontsize=14, fontweight='bold')
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Base Model', 'SFT Model'], fontsize=11)
                ax.set_ylim([0, 100])
                if idx == 0:
                    fig.legend(loc='upper right', bbox_to_anchor=(0.98, 0.9), fontsize=10, framealpha=0.9)
                ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = os.path.join(self.args.vis_dir, f'type_ratio_comparison_{data_type}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Type ratio comparison plot saved: {save_path}")

    def _plot_topk_distribution_comparison(self, results: Dict):
        """
        Plots the Top-K distribution comparison.
        Shows the change in the ratio of each token type in the Top-K candidates.
        """
        token_types = ["Instruction", "Question", "Politeness", "Structural", "Modal"]

        for data_type in self.args.data_types:
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            fig.suptitle(f'{data_type} Dataset - Top-{self.args.top_k} Candidate Token Distribution (Base vs SFT)\n'
                         f'Shows the proportion of each type in the model\'s candidate list.',
                         fontsize=16, fontweight='bold')

            for idx, model_scale in enumerate(self.args.model_scales):
                ax = axes[idx // 2, idx % 2]

                if model_scale not in results or data_type not in results[model_scale]:
                    ax.text(0.5, 0.5, 'Data Missing', ha='center', va='center', fontsize=14)
                    ax.set_title(f'{model_scale} Model', fontsize=14)
                    continue

                base_data = results[model_scale][data_type]["base"]["top_k_dist_summary"]
                sft_data = results[model_scale][data_type]["sft"]["top_k_dist_summary"]

                base_topk_ratios = [base_data.get(t, {}).get("avg_ratio_in_top_k", 0) * 100 for t in token_types]
                sft_topk_ratios = [sft_data.get(t, {}).get("avg_ratio_in_top_k", 0) * 100 for t in token_types]

                x = np.arange(len(token_types))
                width = 0.35

                bars1 = ax.bar(x - width/2, base_topk_ratios, width, label='Base Model',
                               alpha=0.8, color='mediumpurple', edgecolor='black', linewidth=0.5)
                bars2 = ax.bar(x + width/2, sft_topk_ratios, width, label='SFT Model',
                               alpha=0.8, color='gold', edgecolor='black', linewidth=0.5)

                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.annotate(f'{height:.1f}%',
                                        xy=(bar.get_x() + bar.get_width() / 2, height),
                                        xytext=(0, 3), textcoords="offset points",
                                        ha='center', va='bottom', fontsize=9)

                ax.set_xlabel('Token Type', fontsize=12, fontweight='bold')
                ax.set_ylabel(f'Percentage in Top-{self.args.top_k} (%)', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_scale} Model', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(token_types, rotation=45, ha='right')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = os.path.join(self.args.vis_dir, f'topk_distribution_comparison_{data_type}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Top-K distribution plot saved: {save_path}")

    def _plot_first_instruct_rank_comparison(self, results: Dict):
        """
        Plots the first instruction word rank comparison.
        Compares the average rank of the first generated instruction word.
        """
        for data_type in self.args.data_types:
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            fig.suptitle(f'{data_type} Dataset - First Instruction Word Comparison (Base vs SFT)\n'
                         f'A smaller rank means the instruction word is higher in the candidate list.',
                         fontsize=16, fontweight='bold')

            for idx, model_scale in enumerate(self.args.model_scales):
                ax = axes[idx // 2, idx % 2]

                if model_scale not in results or data_type not in results[model_scale]:
                    ax.text(0.5, 0.5, 'Data Missing', ha='center', va='center', fontsize=14)
                    ax.set_title(f'{model_scale} Model', fontsize=14)
                    continue

                base_first = results[model_scale][data_type]["base"]["first_instruct_summary"]
                sft_first = results[model_scale][data_type]["sft"]["first_instruct_summary"]

                metrics = ['Occurrence Rate', 'Avg Position', 'Avg Rank', 'Avg Probability']
                base_values = [
                    base_first.get("occurrence_rate", 0) * 100,
                    base_first.get("avg_position", 0) or 0,
                    base_first.get("avg_rank", 0) or 0,
                    (base_first.get("avg_probability", 0) or 0) * 100
                ]
                sft_values = [
                    sft_first.get("occurrence_rate", 0) * 100,
                    sft_first.get("avg_position", 0) or 0,
                    sft_first.get("avg_rank", 0) or 0,
                    (sft_first.get("avg_probability", 0) or 0) * 100
                ]

                x = np.arange(len(metrics))
                width = 0.35

                bars1 = ax.bar(x - width/2, base_values, width, label='Base Model',
                               alpha=0.8, color='lightcoral', edgecolor='black', linewidth=0.5)
                bars2 = ax.bar(x + width/2, sft_values, width, label='SFT Model',
                               alpha=0.8, color='lightseagreen', edgecolor='black', linewidth=0.5)

                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.annotate(f'{height:.1f}',
                                        xy=(bar.get_x() + bar.get_width() / 2, height),
                                        xytext=(0, 3), textcoords="offset points",
                                        ha='center', va='bottom', fontsize=9)

                ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
                ax.set_ylabel('Value', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_scale} Model', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(metrics, rotation=45, ha='right')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = os.path.join(self.args.vis_dir, f'first_instruct_rank_comparison_{data_type}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ First instruction word plot saved: {save_path}")

    def print_summary(self, comparison_results: Dict):
        """
        Prints a summary of the statistics.

        Input:
            comparison_results: The dictionary of comparison results.
        """
        print("\n" + "="*80)
        print("Analysis Results Summary")
        print("="*80 + "\n")

        for model_scale in self.args.model_scales:
            print(f"\n{'━'*80}")
            print(f"【{model_scale} Model】")
            print(f"{'━'*80}")

            for data_type in self.args.data_types:
                if model_scale not in comparison_results or data_type not in comparison_results[model_scale]:
                    continue

                print(f"\n  📊 Data Category: {data_type}")
                print(f"  {'-'*70}")

                base_stats = comparison_results[model_scale][data_type]["base"]["token_type_summary"]
                sft_stats = comparison_results[model_scale][data_type]["sft"]["token_type_summary"]

                # Print header
                print(f"\n  {'Token Type':<12} {'Base Entropy':<14} {'SFT Entropy':<13} {'Entropy Δ':<12} "
                      f"{'Base Prob':<12} {'SFT Prob':<12} {'Prob Δ':<12}")
                print("  " + "-"*90)

                # Print stats for each token type
                for token_type in ["Instruction", "Question", "Politeness", "Structural", "Modal"]:
                    if token_type in base_stats and token_type in sft_stats:
                        base_entropy = base_stats[token_type]["avg_entropy"]
                        sft_entropy = sft_stats[token_type]["avg_entropy"]
                        entropy_change = ((sft_entropy - base_entropy) / base_entropy * 100) if base_entropy > 0 else 0

                        base_prob = base_stats[token_type]["avg_probability"]
                        sft_prob = sft_stats[token_type]["avg_probability"]
                        prob_change = ((sft_prob - base_prob) / base_prob * 100) if base_prob > 0 else 0

                        print(f"  {token_type:<12} {base_entropy:<14.4f} {sft_entropy:<13.4f} "
                              f"{entropy_change:>+11.2f}% {base_prob:<12.4f} {sft_prob:<12.4f} "
                              f"{prob_change:>+11.2f}%")

                # Print first instruction word stats
                base_first = comparison_results[model_scale][data_type]["base"]["first_instruct_summary"]
                sft_first = comparison_results[model_scale][data_type]["sft"]["first_instruct_summary"]

                print(f"\n  【First Instruction Word Stats】")
                print(f"    Base - Occur Rate: {base_first.get('occurrence_rate', 0)*100:.1f}%, "
                      f"Avg Position: {base_first.get('avg_position', 0):.1f}, "
                      f"Avg Rank: {base_first.get('avg_rank', 0):.1f}")
                print(f"    SFT  - Occur Rate: {sft_first.get('occurrence_rate', 0)*100:.1f}%, "
                      f"Avg Position: {sft_first.get('avg_position', 0):.1f}, "
                      f"Avg Rank: {sft_first.get('avg_rank', 0):.1f}")

        print("\n" + "="*80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
    description='分析base和SFT模型的token概率分布特征',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
    示例用法:
    python token_analysis.py
    python token_analysis.py --model_scales 1B 7B --data_types stackexchange wiki-fact
    python token_analysis.py --top_k 20 --prefix_length 16
    """
    )

    # Input/output paths
    parser.add_argument('--input_dir', type=str, default='/root/autodl-tmp/ift_memorization/results/exp1_generation_16', help='模型生成内容的输入目录')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/ift_memorization/results/exp1_2/visualizations', help='分析结果保存目录')
    parser.add_argument('--vis_dir', type=str, default='/root/autodl-tmp/ift_memorization/results/exp1_2/visualizations', help='可视化图表保存目录')
    # Model configuration
    parser.add_argument('--model_scales', type=str, nargs='+', default=['1B', '7B', ],  help='要分析的模型规模列表')
    # Data types (based on filenames in Attachment 2)
    parser.add_argument('--data_types', type=str, nargs='+',  default=['stackexchange', 'wiki-fact', 'dclm-privacy'], help='要分析的数据类型列表')
    # Data parameters
    parser.add_argument('--prefix_length', type=int, default=16, help='前缀长度（默认16）')
    parser.add_argument('--num_samples', type=int, default=100, help='样本数量（默认50）')
    # Analysis parameters
    parser.add_argument('--top_k', type=int, default=10,  help='分析Top-K个最高概率的token（默认10）')

    return parser.parse_args()


def main():
    """
    Main function.
    """
    args = parse_args()

    # Print configuration
    print("="*80)
    print("Instruction Token Probability Distribution Analysis Tool")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Input Directory:      {args.input_dir}")
    print(f"  Output Directory:     {args.output_dir}")
    print(f"  Visualization Dir:    {args.vis_dir}")
    print(f"  Model Scales:         {', '.join(args.model_scales)}")
    print(f"  Data Types:           {', '.join(args.data_types)}")
    print(f"  Prefix Length:        {args.prefix_length}")
    print(f"  Number of Samples:    {args.num_samples}")
    print(f"  Top-K for Analysis:   {args.top_k}")
    print("\n")

    # Create analyzer
    analyzer = InstructionTokenAnalyzer(args)

    # Run the complete comparison analysis
    try:
        analyzer.compare_base_sft()

        print("\n" + "="*80)
        print("✅ Analysis complete!")
        print("="*80)
        print(f"\nResult files:")
        print(f"  - Detailed analysis: {args.output_dir}/")
        print(f"  - Visualization charts: {args.vis_dir}/")
        print(f"  - Complete comparison: {os.path.join(args.output_dir, 'complete_comparison.json')}")
        print("\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
