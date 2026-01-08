"""
Analyze characteristics of instruction fine-tuning data (updated version)
Includes an updated instruction-verb list
"""
import json
import os
import re
import argparse
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class InstructionAnalyzer:
    def __init__(self, instruction_verbs: Set[str] = None):
        """
        Initialize the analyzer

        Args:
            instruction_verbs: A set of instruction verbs; if None, use the default set
        """
        self.analysis_functions = []

        # Set the instruction verb set
        if instruction_verbs is None:
            # Default instruction verbs (merged from the user-provided lists)
            instruct1 = [
                "translate", "explain", "summarize", "retrieve",
                "revise", 'generate', 'describe', 'classify', 'create',
                "evaluate", "correct", "develop",
                "identify", "analyze", "compose", "demonstrate", "interpret",
                "design", "solve", "follow", "clarify", "say", "help", "act",
                "recommend", "estimate", "edit", "format", "repeat"
            ]

            instruct2 = [
                "write", "give", "find", "create", "make", "describe", "design",
                "generate", "classify", "have", "explain", "tell", "identify",
                "output", "predict", "detect"
            ]

            # Merge and deduplicate (lowercased)
            self.instruction_verbs = set(word.lower() for word in instruct1 + instruct2)
        else:
            self.instruction_verbs = set(word.lower() for word in instruction_verbs)

        print(f"已加载 {len(self.instruction_verbs)} 个指令动词")

    def add_analysis(self, func, name: str):
        """
        Add an analysis function

        Args:
            func: Analysis function
            name: Analysis name
        """
        self.analysis_functions.append((func, name))

    def count_words(self, text: str) -> int:
        """
        Fast word count as a tokenizer substitute

        Args:
            text: Input text

        Returns:
            Word count
        """
        # Simple word count by splitting on whitespace
        return len(text.split())

    def count_characters(self, text: str) -> int:
        """
        Character count

        Args:
            text: Input text

        Returns:
            Character count
        """
        return len(text.strip())

    def analyze_length_distribution(self, data: List[Dict]) -> Dict:
        """
        Analyze length distribution

        Purpose:
        - Understand typical lengths of instructions and responses
        - Provide reference for later pretraining-data filtering

        Args:
            data: List of data items

        Returns:
            A dict of length-distribution statistics
        """
        instruction_word_lengths = []
        response_word_lengths = []
        total_word_lengths = []

        instruction_char_lengths = []
        response_char_lengths = []
        total_char_lengths = []

        for item in data:
            # Handle different data formats
            if 'messages' in item:
                # Handle messages format
                instruction_text = ""
                response_text = ""
                for msg in item['messages']:
                    if msg['role'] == 'user':
                        instruction_text += msg['content'] + " "
                    elif msg['role'] == 'assistant':
                        response_text += msg['content'] + " "
            else:
                # Handle direct format
                instruction_text = item.get("instruction", item.get("instruction_text", ""))
                response_text = item.get("response", item.get("response_text", ""))

            # Word counting
            inst_words = self.count_words(instruction_text)
            resp_words = self.count_words(response_text)

            # Character counting
            inst_chars = self.count_characters(instruction_text)
            resp_chars = self.count_characters(response_text)

            instruction_word_lengths.append(inst_words)
            response_word_lengths.append(resp_words)
            total_word_lengths.append(inst_words + resp_words)

            instruction_char_lengths.append(inst_chars)
            response_char_lengths.append(resp_chars)
            total_char_lengths.append(inst_chars + resp_chars)

        return {
            "words": {
                "instruction": {
                    "mean": np.mean(instruction_word_lengths),
                    "median": np.median(instruction_word_lengths),
                    "std": np.std(instruction_word_lengths),
                    "percentiles": np.percentile(instruction_word_lengths, [10, 25, 50, 75, 90]).tolist(),
                    "max": np.max(instruction_word_lengths),
                    "min": np.min(instruction_word_lengths)
                },
                "response": {
                    "mean": np.mean(response_word_lengths),
                    "median": np.median(response_word_lengths),
                    "std": np.std(response_word_lengths),
                    "percentiles": np.percentile(response_word_lengths, [10, 25, 50, 75, 90]).tolist(),
                    "max": np.max(response_word_lengths),
                    "min": np.min(response_word_lengths)
                },
                "total": {
                    "mean": np.mean(total_word_lengths),
                    "median": np.median(total_word_lengths),
                    "percentiles": np.percentile(total_word_lengths, [10, 25, 50, 75, 90]).tolist()
                }
            },
            "characters": {
                "instruction": {
                    "mean": np.mean(instruction_char_lengths),
                    "median": np.median(instruction_char_lengths),
                    "std": np.std(instruction_char_lengths),
                    "percentiles": np.percentile(instruction_char_lengths, [10, 25, 50, 75, 90]).tolist()
                },
                "response": {
                    "mean": np.mean(response_char_lengths),
                    "median": np.median(response_char_lengths),
                    "std": np.std(response_char_lengths),
                    "percentiles": np.percentile(response_char_lengths, [10, 25, 50, 75, 90]).tolist()
                }
            },
            "raw_data": {
                "instruction_words": instruction_word_lengths,
                "response_words": response_word_lengths,
                "instruction_chars": instruction_char_lengths,
                "response_chars": response_char_lengths
            }
        }

    def extract_text_from_item(self, item: Dict) -> Tuple[str, str]:
        """
        Extract instruction and response text from a data item

        Args:
            item: Data item dict

        Returns:
            (instruction, response) tuple
        """
        if 'messages' in item:
            # Handle messages format
            instruction_text = ""
            response_text = ""
            for msg in item['messages']:
                if msg['role'] == 'user':
                    instruction_text += msg['content'] + " "
                elif msg['role'] == 'assistant':
                    response_text += msg['content'] + " "
        else:
            # Handle direct format
            instruction_text = item.get("instruction", item.get("instruction_text", ""))
            response_text = item.get("response", item.get("response_text", ""))

        return instruction_text.strip(), response_text.strip()

    def analyze_instruction_patterns(self, data: List[Dict]) -> Dict:
        """
        Analyze instruction patterns

        Args:
            data: List of data items

        Returns:
            A dict of instruction-pattern statistics
        """
        # Count instruction starting words
        start_words = Counter()

        # Instruction-type patterns
        patterns = {
            "question": 0,  # Interrogative
            "command": 0,  # Imperative
            "completion": 0,  # Completion task
            "generation": 0,  # Generation task
            "explanation": 0,  # Explanation task
            "translation": 0,  # Translation task
            "summarization": 0,  # Summarization task
            "analysis": 0,  # Analysis task
            "coding": 0,  # Coding task
            "math": 0,  # Math task
        }

        # Common instruction verbs (use the verb set configured in the class)
        instruction_verbs_counter = Counter()

        for item in data:
            instruction, _ = self.extract_text_from_item(item)
            instruction = instruction.lower()

            if not instruction:
                continue

            # Count starting words
            words = instruction.split()
            if words:
                start_words[words[0]] += 1

            # Identify instruction types
            if "?" in instruction or any(q in instruction for q in ["what", "why", "how", "when", "where", "who"]):
                patterns["question"] += 1

            if any(cmd in instruction for cmd in ["write", "create", "generate", "make", "produce"]):
                patterns["generation"] += 1

            if any(exp in instruction for exp in ["explain", "describe", "elaborate", "define"]):
                patterns["explanation"] += 1

            if "complete" in instruction or "continue" in instruction or "finish" in instruction:
                patterns["completion"] += 1

            if "translate" in instruction or "translation" in instruction:
                patterns["translation"] += 1

            if any(sum_word in instruction for sum_word in ["summarize", "summary", "brief", "outline"]):
                patterns["summarization"] += 1

            if any(ana in instruction for ana in ["analyze", "analysis", "evaluate", "assess", "compare"]):
                patterns["analysis"] += 1

            if any(code in instruction for code in ["code", "function", "program", "script", "algorithm"]):
                patterns["coding"] += 1

            if any(math in instruction for math in ["calculate", "solve", "equation", "formula", "math"]):
                patterns["math"] += 1

            # Extract verbs (using self.instruction_verbs)
            for word in words:
                if word in self.instruction_verbs:
                    instruction_verbs_counter[word] += 1

        # Convert to percentages
        total = len(data)
        patterns_pct = {k: (v / total) * 100 for k, v in patterns.items()}

        return {
            "start_words": dict(start_words.most_common(20)),
            "patterns": patterns_pct,
            "instruction_verbs": dict(instruction_verbs_counter.most_common(20))
        }

    def analyze_response_patterns(self, data: List[Dict]) -> Dict:
        """
        Analyze response patterns

        Args:
            data: List of data items

        Returns:
            A dict of response-pattern statistics
        """
        response_formats = {
            "numbered_list": 0,  # Numbered list
            "bullet_points": 0,  # Bullet points
            "step_by_step": 0,  # Step-by-step
            "code_block": 0,  # Code block
            "single_paragraph": 0,  # Single paragraph
            "multi_paragraph": 0,  # Multiple paragraphs
            "here_is": 0,  # Starts with "Here is"
            "conversational": 0,  # Conversational
            "structured": 0,  # Structured response
        }

        # Response starting phrases
        start_phrases = Counter()

        for item in data:
            _, response = self.extract_text_from_item(item)
            if not response:
                continue

            response_lower = response.lower()

            # Check formats
            if re.search(r'^\d+\.', response, re.MULTILINE):
                response_formats["numbered_list"] += 1

            if re.search(r'^[\*\-\•]', response, re.MULTILINE):
                response_formats["bullet_points"] += 1

            if re.search(r'step \d|first.*then|next.*step', response_lower):
                response_formats["step_by_step"] += 1

            if "```" in response or re.search(r'`[^`]+`', response):
                response_formats["code_block"] += 1

            if (response_lower.startswith("here is") or response_lower.startswith("here's") or
                response_lower.startswith("here are")):
                response_formats["here_is"] += 1

            if any(conv in response_lower for conv in ["i think", "in my opinion", "i believe", "i would"]):
                response_formats["conversational"] += 1

            if re.search(r'(first|second|third|finally|in conclusion|therefore)', response_lower):
                response_formats["structured"] += 1

            # Count paragraphs
            paragraphs = [p for p in response.split('\n\n') if p.strip()]
            if len(paragraphs) == 1:
                response_formats["single_paragraph"] += 1
            elif len(paragraphs) > 1:
                response_formats["multi_paragraph"] += 1

            # Extract starting phrases (first 5 words)
            words = response.split()[:5]
            if len(words) >= 2:
                phrase = " ".join(words[:2]).lower()
                start_phrases[phrase] += 1

        # Convert to percentages
        total = len(data)
        formats_pct = {k: (v / total) * 100 for k, v in response_formats.items()}

        return {
            "formats": formats_pct,
            "start_phrases": dict(start_phrases.most_common(20))
        }

    def analyze_topic_distribution(self, data: List[Dict]) -> Dict:
        """
        Analyze topic distribution

        Args:
            data: List of data items

        Returns:
            Topic distribution dict (in percentages)
        """
        # More detailed topic keyword matching
        topics = {
            "编程开发": ["code", "function", "program", "python", "javascript", "algorithm", "software", "debug", "api"],
            "数学计算": ["calculate", "solve", "equation", "number", "mathematical", "formula", "statistics"],
            "科学研究": ["scientific", "research", "experiment", "hypothesis", "theory", "data", "analysis"],
            "写作创作": ["write", "essay", "story", "paragraph", "article", "creative", "poem"],
            "语言学习": ["translate", "grammar", "sentence", "word", "language", "english", "chinese"],
            "常识问答": ["explain", "what is", "define", "describe", "general knowledge"],
            "商业金融": ["business", "market", "finance", "economic", "investment", "strategy"],
            "教育学习": ["learn", "study", "education", "teach", "tutorial", "lesson"],
            "技术支持": ["help", "how to", "troubleshoot", "fix", "setup", "configure"],
            "娱乐休闲": ["game", "movie", "music", "entertainment", "fun", "hobby"]
        }

        topic_counts = defaultdict(int)

        for item in data:
            instruction, response = self.extract_text_from_item(item)
            text = (instruction + " " + response).lower()

            for topic, keywords in topics.items():
                if any(keyword in text for keyword in keywords):
                    topic_counts[topic] += 1

        # Convert to percentages
        total = len(data)
        topic_pct = {k: (v / total) * 100 for k, v in topic_counts.items()}

        return topic_pct

    def create_visualizations(self, results: Dict, output_dir: str):
        """
        Create visualization plots

        Args:
            results: Analysis results dict
            output_dir: Output directory path
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Set plot style
        plt.style.use('default')
        fig_size = (12, 8)

        # 1. Length distribution histograms
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Instruction word-count distribution
        inst_words = results['length_distribution']['raw_data']['instruction_words']
        ax1.hist(inst_words, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('指令长度分布 (词数)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('词数')
        ax1.set_ylabel('频次')
        ax1.axvline(np.mean(inst_words), color='red', linestyle='--', label=f'平均值: {np.mean(inst_words):.1f}')
        ax1.legend()

        # Response word-count distribution
        resp_words = results['length_distribution']['raw_data']['response_words']
        ax2.hist(resp_words, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_title('回答长度分布 (词数)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('词数')
        ax2.set_ylabel('频次')
        ax2.axvline(np.mean(resp_words), color='red', linestyle='--', label=f'平均值: {np.mean(resp_words):.1f}')
        ax2.legend()

        # Instruction character-count distribution
        inst_chars = results['length_distribution']['raw_data']['instruction_chars']
        ax3.hist(inst_chars, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_title('指令长度分布 (字符数)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('字符数')
        ax3.set_ylabel('频次')
        ax3.axvline(np.mean(inst_chars), color='red', linestyle='--', label=f'平均值: {np.mean(inst_chars):.1f}')
        ax3.legend()

        # Response character-count distribution
        resp_chars = results['length_distribution']['raw_data']['response_chars']
        ax4.hist(resp_chars, bins=50, alpha=0.7, color='pink', edgecolor='black')
        ax4.set_title('回答长度分布 (字符数)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('字符数')
        ax4.set_ylabel('频次')
        ax4.axvline(np.mean(resp_chars), color='red', linestyle='--', label=f'平均值: {np.mean(resp_chars):.1f}')
        ax4.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'length_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Instruction-type distribution pie chart
        patterns = results['instruction_patterns']['patterns']
        patterns_filtered = {k: v for k, v in patterns.items() if v > 1}  # Only show >1%

        fig, ax = plt.subplots(figsize=fig_size)
        colors = plt.cm.Set3(np.linspace(0, 1, len(patterns_filtered)))
        wedges, texts, autotexts = ax.pie(patterns_filtered.values(),
                                         labels=patterns_filtered.keys(),
                                         autopct='%1.1f%%',
                                         colors=colors,
                                         startangle=90)
        ax.set_title('指令类型分布', fontsize=16, fontweight='bold')
        plt.setp(autotexts, size=10, weight="bold")
        plt.savefig(os.path.join(output_dir, 'instruction_patterns.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Response-format distribution bar chart
        formats = results['response_patterns']['formats']
        formats_filtered = {k: v for k, v in formats.items() if v > 1}

        fig, ax = plt.subplots(figsize=fig_size)
        bars = ax.barh(list(formats_filtered.keys()), list(formats_filtered.values()),
                       color='lightcoral', edgecolor='black')
        ax.set_title('回答格式分布', fontsize=16, fontweight='bold')
        ax.set_xlabel('百分比 (%)')

        # Add value labels on the bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.1f}%', ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'response_formats.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Topic distribution bar chart
        topics = results['topic_distribution']
        topics_filtered = {k: v for k, v in topics.items() if v > 0.5}

        fig, ax = plt.subplots(figsize=fig_size)
        bars = ax.bar(list(topics_filtered.keys()), list(topics_filtered.values()),
                      color='lightblue', edgecolor='black')
        ax.set_title('主题分布', fontsize=16, fontweight='bold')
        ax.set_ylabel('百分比 (%)')
        ax.set_xlabel('主题类别')
        plt.xticks(rotation=45, ha='right')

        # Add value labels on the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'topic_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Bar chart styled like a word-cloud for common instruction verbs
        verbs = results['instruction_patterns']['instruction_verbs']
        top_verbs = dict(list(verbs.items())[:15])

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(list(top_verbs.keys()), list(top_verbs.values()),
                       color='gold', edgecolor='black')
        ax.set_title('最常用指令动词 (Top 15)', fontsize=16, fontweight='bold')
        ax.set_xlabel('出现次数')

        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{int(width)}', ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'instruction_verbs.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"所有可视化图表已保存到: {output_dir}")

    def run_analysis(self, data: List[Dict]) -> Dict:
        """
        Run all analyses

        Args:
            data: List of data items

        Returns:
            Analysis results dict
        """
        results = {}

        # Add all analysis functions
        self.add_analysis(self.analyze_length_distribution, "length_distribution")
        self.add_analysis(self.analyze_instruction_patterns, "instruction_patterns")
        self.add_analysis(self.analyze_response_patterns, "response_patterns")
        self.add_analysis(self.analyze_topic_distribution, "topic_distribution")

        # Execute analyses
        for func, name in self.analysis_functions:
            print(f"执行分析: {name}")
            results[name] = func(data)

        return results


def main():
    """
    Main function: parse arguments and run analysis
    """
    parser = argparse.ArgumentParser(description="分析指令微调数据")
    parser.add_argument("--input_file", type=str,
                        default="/root/autodl-tmp/ift_memorization/data/instruction_test_data/olmo_instruction_tulu3_intersection.jsonl",
                        help="输入文件路径")
    parser.add_argument("--output_dir", type=str,
                        default="/root/autodl-tmp/ift_memorization/data/instruction_test_data/analysis",
                        help="输出目录")
    parser.add_argument("--max_samples", type=int,
                        default=None,
                        help="最大分析样本数")

    args = parser.parse_args()

    # Check whether the input file exists
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件不存在: {args.input_file}")
        print("请检查文件路径或者使用 --input_file 指定正确的路径")
        return

    # Load data
    print(f"加载数据: {args.input_file}")
    data = []
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line))
                    if args.max_samples and len(data) >= args.max_samples:
                        break
                except json.JSONDecodeError:
                    print(f"警告: 第{line_num}行JSON格式错误，跳过")
                    continue
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    print(f"成功加载 {len(data)} 条数据")

    if len(data) == 0:
        print("错误: 没有有效数据可以分析")
        return

    # Create analyzer (using the updated instruction-verb list)
    analyzer = InstructionAnalyzer()

    # Run analyses
    results = analyzer.run_analysis(data)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save results
    output_path = os.path.join(args.output_dir, "instruction_analysis.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"分析结果保存到: {output_path}")

    # Create visualizations
    print("创建可视化图表...")
    analyzer.create_visualizations(results, args.output_dir)

    # Print detailed summary
    print("\n" + "="*60)
    print("📊 指令微调数据集分析报告")
    print("="*60)

    print(f"\n📈 基本统计信息:")
    print(f"  总数据量: {len(data):,} 条")
    print(f"  指令平均长度: {results['length_distribution']['words']['instruction']['mean']:.1f} 词")
    print(f"  回答平均长度: {results['length_distribution']['words']['response']['mean']:.1f} 词")
    print(f"  指令平均字符数: {results['length_distribution']['characters']['instruction']['mean']:.0f}")
    print(f"  回答平均字符数: {results['length_distribution']['characters']['response']['mean']:.0f}")

    print(f"\n🎯 指令类型分布 (>5%):")
    for pattern, pct in results['instruction_patterns']['patterns'].items():
        if pct > 5:
            print(f"  {pattern}: {pct:.1f}%")

    print(f"\n📝 回答格式分布 (>5%):")
    for format_type, pct in results['response_patterns']['formats'].items():
        if pct > 5:
            print(f"  {format_type}: {pct:.1f}%")

    print(f"\n🏷️ 主题分布 (>2%):")
    for topic, pct in results['topic_distribution'].items():
        if pct > 2:
            print(f"  {topic}: {pct:.1f}%")

    print(f"\n🔤 最常用指令动词 (Top 10):")
    for i, (verb, count) in enumerate(list(results['instruction_patterns']['instruction_verbs'].items())[:10], 1):
        print(f"  {i:2d}. {verb}: {count} 次")

    print(f"\n💬 常见回答开头 (Top 5):")
    for i, (phrase, count) in enumerate(list(results['response_patterns']['start_phrases'].items())[:5], 1):
        print(f"  {i}. \"{phrase}\": {count} 次")

    print(f"\n📊 所有图表已保存到: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
