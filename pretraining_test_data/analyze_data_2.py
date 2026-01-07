"""
分析下载的数据集特征（优化版本）
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict, Any
import argparse
import re
import seaborn as sns
from tqdm import tqdm


class DataAnalyzer:
    def __init__(self, max_words: int = 512):
        """
        初始化分析器

        Args:
            max_words: 每条数据最多处理的词数
        """
        self.max_words = max_words

    def load_data(self, file_path: str) -> List[Dict]:
        """
        加载JSONL格式数据，显示进度，并处理可能存在的格式错误。

        **--- 修改部分 ---**
        增加了try-except块来捕获JSON解析错误，使程序不会因单行数据损坏而崩溃。
        """
        data = []
        malformed_lines = 0  # 增加一个计数器，用于记录格式错误的行数

        # 先统计总行数以便tqdm正确显示进度
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                total_lines = sum(1 for _ in f)
        except FileNotFoundError:
            print(f"错误：文件未找到 {file_path}")
            return []

        # 带进度条加载数据
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f, total=total_lines, desc="加载数据")):
                try:
                    # 尝试解析当前行
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    # 如果解析失败，则跳过这一行，并增加计数器
                    malformed_lines += 1
                    # 可以选择性地打印一些错误信息来帮助调试，但对于大量错误，这会刷屏
                    # if malformed_lines < 5: # 只打印前几个错误
                    #     print(f"警告：跳过格式错误的第 {i+1} 行。内容：{line.strip()}")

        if malformed_lines > 0:
            print(f"\n警告：加载完成。共发现并跳过了 {malformed_lines} 行格式错误的数据。")

        return data

    def analyze_length_distribution(self, texts: List[str]) -> Dict:
        """
        分析文本长度分布（基于word count，快速版本）

        为什么做这个分析：
        - 了解数据的长度特征，用于后续分段策略
        - 确定合适的context length划分
        """
        char_lengths = []
        word_lengths = []

        # 使用tqdm显示进度
        for text in tqdm(texts, desc="分析长度分布"):
            # 只处理前max_words个词
            words = text.split()[:self.max_words]

            # 字符长度
            truncated_text = ' '.join(words)
            char_lengths.append(len(truncated_text))

            # 词数
            word_lengths.append(len(words))

        return {
            "char_stats": {
                "mean": np.mean(char_lengths),
                "median": np.median(char_lengths),
                "std": np.std(char_lengths),
                "min": np.min(char_lengths),
                "max": np.max(char_lengths),
                "quantiles": np.percentile(char_lengths, [25, 50, 75, 90, 95, 99])
            },
            "word_stats": {
                "mean": np.mean(word_lengths),
                "median": np.median(word_lengths),
                "std": np.std(word_lengths),
                "min": np.min(word_lengths),
                "max": np.max(word_lengths),
                "quantiles": np.percentile(word_lengths, [25, 50, 75, 90, 95, 99])
            },
            "char_lengths": char_lengths,
            "word_lengths": word_lengths
        }

    def analyze_text_patterns(self, texts: List[str]) -> Dict:
        """
        分析文本模式（优化版本，只处理前512词）

        为什么做这个分析：
        - 识别指令格式特征（问题、步骤等）
        - 了解数据的结构化程度
        """
        patterns = {
            "has_question": 0,  # 包含问句
            "has_steps": 0,  # 包含步骤格式
            "has_list": 0,  # 包含列表
            "has_code": 0,  # 包含代码
            "has_url": 0,  # 包含URL
            "language_mix": 0,  # 中英混合
        }

        # 预编译正则表达式以提高性能
        question_pattern = re.compile(r'\?|how |what |why |when |where |who ', re.IGNORECASE)
        steps_pattern = re.compile(r'step \d|first,|second,|finally|then |next ', re.IGNORECASE)
        list_pattern = re.compile(r'^\s*[\d\-\*]\.\s+', re.MULTILINE)
        code_pattern = re.compile(r'```|def |class |function |import |print\(')
        url_pattern = re.compile(r'https?://|www\.')
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')

        for text in tqdm(texts, desc="分析文本模式"):
            # 只处理前max_words个词
            words = text.split()[:self.max_words]
            truncated_text = ' '.join(words)

            # 检测各种模式
            if question_pattern.search(truncated_text):
                patterns["has_question"] += 1
            if steps_pattern.search(truncated_text):
                patterns["has_steps"] += 1
            if list_pattern.search(truncated_text):
                patterns["has_list"] += 1
            if code_pattern.search(truncated_text):
                patterns["has_code"] += 1
            if url_pattern.search(truncated_text):
                patterns["has_url"] += 1
            if chinese_pattern.search(truncated_text):
                patterns["language_mix"] += 1

        # 转换为百分比
        total = len(texts)
        patterns_pct = {k: (v / total) * 100 for k, v in patterns.items()}

        return patterns_pct

    def analyze_vocabulary(self, texts: List[str], top_k: int = 100) -> Dict:
        """
        分析词汇特征（基于word分割，快速版本）

        为什么做这个分析：
        - 了解数据的词汇多样性
        - 识别高频词和特殊词汇
        """
        all_words = []
        # 只采样前1000条以节省时间
        sample_size = max(1000, len(texts))

        for text in tqdm(texts[:sample_size], desc="分析词汇"):
            # 只处理前max_words个词
            words = text.split()[:self.max_words]
            # 转小写以统一计数
            all_words.extend([w.lower() for w in words])

        vocab_counter = Counter(all_words)

        return {
            "unique_words": len(vocab_counter),
            "total_words": len(all_words),
            "type_token_ratio": len(vocab_counter) / len(all_words) if all_words else 0,
            "top_words": vocab_counter.most_common(top_k)
        }

    def plot_distributions(self, analysis_results: Dict, output_dir: str):
        """绘制分布图"""
        os.makedirs(output_dir, exist_ok=True)

        # 绘制word长度分布
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.hist(analysis_results["length"]["word_lengths"], bins=50, edgecolor='black')
        plt.xlabel("Word Length")
        plt.ylabel("Frequency")
        plt.title("Word Length Distribution")

        plt.subplot(1, 3, 2)
        plt.hist(np.log10(np.array(analysis_results["length"]["word_lengths"]) + 1),
                 bins=50, edgecolor='black')
        plt.xlabel("Log10(Word Length)")
        plt.ylabel("Frequency")
        plt.title("Log Word Length Distribution")

        plt.subplot(1, 3, 3)
        # 绘制文本模式
        patterns = analysis_results["patterns"]
        plt.bar(range(len(patterns)), list(patterns.values()))
        plt.xticks(range(len(patterns)), list(patterns.keys()), rotation=45, ha='right')
        plt.ylabel("Percentage (%)")
        plt.title("Text Patterns")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{analysis_results['dataset_name']}_analysis.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="分析预训练语料数据")
    parser.add_argument("--data_dir", type=str, default="../../data/pretraining_test_data", help="数据目录")
    parser.add_argument("--output_dir", type=str, default="../../data/pretraining_test_data/analysis", help="分析结果输出目录")
    parser.add_argument("--max_words", type=int, default=512, help="每条数据最多处理的词数")
    parser.add_argument("--datasets", nargs="+", default=["dclm"],
                        # default=["stackexchange", "wiki", "dclm"],
                        help="要处理的数据集列表")

    args = parser.parse_args()

    analyzer = DataAnalyzer(max_words=args.max_words)

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"分析 {dataset_name} 数据集...")
        print(f"{'='*60}")

        file_path = os.path.join(args.data_dir, f"{dataset_name}_top5M.jsonl")
        data = analyzer.load_data(file_path)
        texts = [item["text"] for item in data]

        # 执行分析
        print("\n开始分析...")
        length_analysis = analyzer.analyze_length_distribution(texts[:1000000])
        pattern_analysis = analyzer.analyze_text_patterns(texts[:1000000])
        vocab_analysis = analyzer.analyze_vocabulary(texts[:1000000])

        # 整合结果
        results = {
            "dataset_name": dataset_name,
            "sample_size": len(texts),
            "length": length_analysis,
            "patterns": pattern_analysis,
            "vocabulary": vocab_analysis
        }

        # 打印摘要
        print(f"\n{'='*60}")
        print("分析结果摘要：")
        print(f"{'='*60}")
        print(f"样本数量: {len(texts):,}")
        print(f"平均word长度: {length_analysis['word_stats']['mean']:.1f}")
        print(f"Word长度中位数: {length_analysis['word_stats']['median']:.1f}")
        print(f"Word长度标准差: {length_analysis['word_stats']['std']:.1f}")
        print(f"\nWord长度分位数:")
        for i, q in enumerate([25, 50, 75, 90, 95, 99]):
            print(f"  {q}%: {length_analysis['word_stats']['quantiles'][i]:.0f}")
        print(f"\n包含问句的比例: {pattern_analysis['has_question']:.1f}%")
        print(f"包含步骤格式的比例: {pattern_analysis['has_steps']:.1f}%")
        print(f"包含列表的比例: {pattern_analysis['has_list']:.1f}%")
        print(f"包含代码的比例: {pattern_analysis['has_code']:.1f}%")
        print(f"包含URL的比例: {pattern_analysis['has_url']:.1f}%")
        print(f"中英混合的比例: {pattern_analysis['language_mix']:.1f}%")
        print(f"\n词汇多样性(TTR): {vocab_analysis['type_token_ratio']:.3f}")
        print(f"唯一词数: {vocab_analysis['unique_words']:,}")
        print(f"总词数: {vocab_analysis['total_words']:,}")

        # 保存详细结果
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{dataset_name}_analysis.json")

        def numpy_encoder(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        print(f"\n生成可视化图表...")
        analyzer.plot_distributions(results, args.output_dir)

        print(f"\n保存分析结果...")
        with open(output_path, "w", encoding="utf-8") as f:
            # 移除无法序列化的数据
            results["length"].pop("char_lengths")
            results["length"].pop("word_lengths")
            results["vocabulary"].pop("top_words")
            json.dump(results, f, indent=2, ensure_ascii=False, default=numpy_encoder)

        print(f"✓ 分析完成！结果已保存到: {output_path}")
        print(f"✓ 可视化图表已保存到: {args.output_dir}/{dataset_name}_analysis.png")


if __name__ == "__main__":
    main()