"""
分析指令微调数据的特征（更新版）
包含更新的指令词列表
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class InstructionAnalyzer:
    def __init__(self, instruction_verbs: Set[str] = None):
        """
        初始化分析器

        Args:
            instruction_verbs: 指令动词集合，如果为None则使用默认集合
        """
        self.analysis_functions = []

        # 设置指令动词集合
        if instruction_verbs is None:
            # 默认的指令动词（合并用户提供的列表）
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

            # 合并并去重（转为小写）
            self.instruction_verbs = set(word.lower() for word in instruct1 + instruct2)
        else:
            self.instruction_verbs = set(word.lower() for word in instruction_verbs)

        print(f"已加载 {len(self.instruction_verbs)} 个指令动词")

    def add_analysis(self, func, name: str):
        """
        添加分析函数

        Args:
            func: 分析函数
            name: 分析名称
        """
        self.analysis_functions.append((func, name))

    def count_words(self, text: str) -> int:
        """
        快速词数统计，替代tokenizer

        Args:
            text: 输入文本

        Returns:
            词数
        """
        # 简单的词数统计，按空格分割
        return len(text.split())

    def count_characters(self, text: str) -> int:
        """
        字符数统计

        Args:
            text: 输入文本

        Returns:
            字符数
        """
        return len(text.strip())

    def analyze_length_distribution(self, data: List[Dict]) -> Dict:
        """
        分析长度分布

        目的：
        - 了解指令和回答的典型长度
        - 为后续筛选预训练数据提供参考

        Args:
            data: 数据列表

        Returns:
            长度分布统计字典
        """
        instruction_word_lengths = []
        response_word_lengths = []
        total_word_lengths = []

        instruction_char_lengths = []
        response_char_lengths = []
        total_char_lengths = []

        for item in data:
            # 处理不同的数据格式
            if 'messages' in item:
                # 处理messages格式
                instruction_text = ""
                response_text = ""
                for msg in item['messages']:
                    if msg['role'] == 'user':
                        instruction_text += msg['content'] + " "
                    elif msg['role'] == 'assistant':
                        response_text += msg['content'] + " "
            else:
                # 处理直接格式
                instruction_text = item.get("instruction", item.get("instruction_text", ""))
                response_text = item.get("response", item.get("response_text", ""))

            # 词数统计
            inst_words = self.count_words(instruction_text)
            resp_words = self.count_words(response_text)

            # 字符数统计
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
        从数据项中提取指令和回答文本

        Args:
            item: 数据项字典

        Returns:
            (instruction, response) 元组
        """
        if 'messages' in item:
            # 处理messages格式
            instruction_text = ""
            response_text = ""
            for msg in item['messages']:
                if msg['role'] == 'user':
                    instruction_text += msg['content'] + " "
                elif msg['role'] == 'assistant':
                    response_text += msg['content'] + " "
        else:
            # 处理直接格式
            instruction_text = item.get("instruction", item.get("instruction_text", ""))
            response_text = item.get("response", item.get("response_text", ""))

        return instruction_text.strip(), response_text.strip()

    def analyze_instruction_patterns(self, data: List[Dict]) -> Dict:
        """
        分析指令模式

        Args:
            data: 数据列表

        Returns:
            指令模式统计字典
        """
        # 指令开头词统计
        start_words = Counter()

        # 指令类型模式
        patterns = {
            "question": 0,  # 疑问句
            "command": 0,  # 命令句
            "completion": 0,  # 补全任务
            "generation": 0,  # 生成任务
            "explanation": 0,  # 解释任务
            "translation": 0,  # 翻译任务
            "summarization": 0,  # 总结任务
            "analysis": 0,  # 分析任务
            "coding": 0,  # 编程任务
            "math": 0,  # 数学任务
        }

        # 常见指令动词（使用类初始化时设置的指令动词集合）
        instruction_verbs_counter = Counter()

        for item in data:
            instruction, _ = self.extract_text_from_item(item)
            instruction = instruction.lower()

            if not instruction:
                continue

            # 统计开头词
            words = instruction.split()
            if words:
                start_words[words[0]] += 1

            # 识别指令类型
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

            # 提取动词（使用self.instruction_verbs）
            for word in words:
                if word in self.instruction_verbs:
                    instruction_verbs_counter[word] += 1

        # 转换为百分比
        total = len(data)
        patterns_pct = {k: (v / total) * 100 for k, v in patterns.items()}

        return {
            "start_words": dict(start_words.most_common(20)),
            "patterns": patterns_pct,
            "instruction_verbs": dict(instruction_verbs_counter.most_common(20))
        }

    def analyze_response_patterns(self, data: List[Dict]) -> Dict:
        """
        分析回答模式

        Args:
            data: 数据列表

        Returns:
            回答模式统计字典
        """
        response_formats = {
            "numbered_list": 0,  # 编号列表
            "bullet_points": 0,  # 项目符号
            "step_by_step": 0,  # 分步骤
            "code_block": 0,  # 代码块
            "single_paragraph": 0,  # 单段落
            "multi_paragraph": 0,  # 多段落
            "here_is": 0,  # "Here is"开头
            "conversational": 0,  # 对话式
            "structured": 0,  # 结构化回答
        }

        # 回答开头短语
        start_phrases = Counter()

        for item in data:
            _, response = self.extract_text_from_item(item)
            if not response:
                continue

            response_lower = response.lower()

            # 检查格式
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

            # 统计段落数
            paragraphs = [p for p in response.split('\n\n') if p.strip()]
            if len(paragraphs) == 1:
                response_formats["single_paragraph"] += 1
            elif len(paragraphs) > 1:
                response_formats["multi_paragraph"] += 1

            # 提取开头短语（前5个词）
            words = response.split()[:5]
            if len(words) >= 2:
                phrase = " ".join(words[:2]).lower()
                start_phrases[phrase] += 1

        # 转换为百分比
        total = len(data)
        formats_pct = {k: (v / total) * 100 for k, v in response_formats.items()}

        return {
            "formats": formats_pct,
            "start_phrases": dict(start_phrases.most_common(20))
        }

    def analyze_topic_distribution(self, data: List[Dict]) -> Dict:
        """
        分析主题分布

        Args:
            data: 数据列表

        Returns:
            主题分布字典（百分比形式）
        """
        # 更详细的主题关键词匹配
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

        # 转换为百分比
        total = len(data)
        topic_pct = {k: (v / total) * 100 for k, v in topic_counts.items()}

        return topic_pct

    def create_visualizations(self, results: Dict, output_dir: str):
        """
        创建可视化图表

        Args:
            results: 分析结果字典
            output_dir: 输出目录路径
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 设置图表样式
        plt.style.use('default')
        fig_size = (12, 8)

        # 1. 长度分布直方图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 指令词数分布
        inst_words = results['length_distribution']['raw_data']['instruction_words']
        ax1.hist(inst_words, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('指令长度分布 (词数)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('词数')
        ax1.set_ylabel('频次')
        ax1.axvline(np.mean(inst_words), color='red', linestyle='--', label=f'平均值: {np.mean(inst_words):.1f}')
        ax1.legend()

        # 回答词数分布
        resp_words = results['length_distribution']['raw_data']['response_words']
        ax2.hist(resp_words, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_title('回答长度分布 (词数)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('词数')
        ax2.set_ylabel('频次')
        ax2.axvline(np.mean(resp_words), color='red', linestyle='--', label=f'平均值: {np.mean(resp_words):.1f}')
        ax2.legend()

        # 指令字符数分布
        inst_chars = results['length_distribution']['raw_data']['instruction_chars']
        ax3.hist(inst_chars, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_title('指令长度分布 (字符数)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('字符数')
        ax3.set_ylabel('频次')
        ax3.axvline(np.mean(inst_chars), color='red', linestyle='--', label=f'平均值: {np.mean(inst_chars):.1f}')
        ax3.legend()

        # 回答字符数分布
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

        # 2. 指令类型分布饼图
        patterns = results['instruction_patterns']['patterns']
        patterns_filtered = {k: v for k, v in patterns.items() if v > 1}  # 只显示大于1%的

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

        # 3. 回答格式分布条形图
        formats = results['response_patterns']['formats']
        formats_filtered = {k: v for k, v in formats.items() if v > 1}

        fig, ax = plt.subplots(figsize=fig_size)
        bars = ax.barh(list(formats_filtered.keys()), list(formats_filtered.values()),
                       color='lightcoral', edgecolor='black')
        ax.set_title('回答格式分布', fontsize=16, fontweight='bold')
        ax.set_xlabel('百分比 (%)')

        # 在条形图上添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.1f}%', ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'response_formats.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. 主题分布条形图
        topics = results['topic_distribution']
        topics_filtered = {k: v for k, v in topics.items() if v > 0.5}

        fig, ax = plt.subplots(figsize=fig_size)
        bars = ax.bar(list(topics_filtered.keys()), list(topics_filtered.values()),
                      color='lightblue', edgecolor='black')
        ax.set_title('主题分布', fontsize=16, fontweight='bold')
        ax.set_ylabel('百分比 (%)')
        ax.set_xlabel('主题类别')
        plt.xticks(rotation=45, ha='right')

        # 在条形图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'topic_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. 常用指令动词词云风格条形图
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
        运行所有分析

        Args:
            data: 数据列表

        Returns:
            分析结果字典
        """
        results = {}

        # 添加所有分析函数
        self.add_analysis(self.analyze_length_distribution, "length_distribution")
        self.add_analysis(self.analyze_instruction_patterns, "instruction_patterns")
        self.add_analysis(self.analyze_response_patterns, "response_patterns")
        self.add_analysis(self.analyze_topic_distribution, "topic_distribution")

        # 执行分析
        for func, name in self.analysis_functions:
            print(f"执行分析: {name}")
            results[name] = func(data)

        return results


def main():
    """
    主函数：解析参数并执行分析
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

    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件不存在: {args.input_file}")
        print("请检查文件路径或者使用 --input_file 指定正确的路径")
        return

    # 加载数据
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

    # 创建分析器（使用更新的指令动词列表）
    analyzer = InstructionAnalyzer()

    # 执行分析
    results = analyzer.run_analysis(data)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存结果
    output_path = os.path.join(args.output_dir, "instruction_analysis.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"分析结果保存到: {output_path}")

    # 创建可视化
    print("创建可视化图表...")
    analyzer.create_visualizations(results, args.output_dir)

    # 打印详细摘要
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