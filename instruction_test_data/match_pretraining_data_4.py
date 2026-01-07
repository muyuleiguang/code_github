"""
使用指令特征匹配和筛选预训练数据
"""
import json
import os
import pickle
import re
import argparse
from typing import List, Dict, Set, Tuple
from tqdm import tqdm
import numpy as np


class PretrainingMatcher:
    def __init__(self, features_path: str):
        """
        初始化匹配器

        Args:
            features_path: 指令特征文件路径
        """
        with open(features_path, "rb") as f:
            self.features = pickle.load(f)

        self.matchers = []  # 存储匹配函数

    def add_matcher(self, func, name: str, weight: float = 1.0):
        """添加匹配函数"""
        self.matchers.append((func, name, weight))

    def match_instruction_format(self, text: str) -> float:
        """
        匹配指令格式

        返回: 匹配分数 (0-1)
        """
        score = 0.0
        text_lower = text.lower()

        # 检查指令动词
        if "instruction_verbs" in self.features:
            verbs = self.features["instruction_verbs"]
            verb_count = sum(1 for verb in verbs if verb in text_lower)
            score += min(verb_count / 3, 1.0) * 0.3  # 最多占30%分数

        # 检查疑问模式
        if "question_patterns" in self.features:
            patterns = self.features["question_patterns"]
            pattern_count = sum(1 for pattern in patterns if pattern in text_lower)
            score += min(pattern_count / 2, 1.0) * 0.2  # 最多占20%分数

        # 检查指令前缀
        if "instruction_prefixes" in self.features:
            prefixes = self.features["instruction_prefixes"]
            has_prefix = any(text_lower.startswith(prefix) for prefix in prefixes)
            score += 0.2 if has_prefix else 0

        # 检查是否包含问号（疑问句特征）
        if "?" in text:
            score += 0.1

        # 检查命令式特征（祈使句）
        imperative_patterns = [r'^(please\s+)?[a-z]+\s+', r'^(can|could|would)\s+you']
        if any(re.match(pattern, text_lower) for pattern in imperative_patterns):
            score += 0.2

        return min(score, 1.0)

    def match_response_format(self, text: str) -> float:
        """
        匹配回答格式

        返回: 匹配分数 (0-1)
        """
        score = 0.0
        text_lower = text.lower()

        # 检查回答模板
        if "response_templates" in self.features:
            templates = self.features["response_templates"]

            for template in templates:
                if isinstance(template, str) and not template.startswith('^'):
                    # 普通字符串模板
                    if template in text_lower[:100]:  # 只检查开头
                        score += 0.3
                        break
                elif isinstance(template, str) and template.startswith('^'):
                    # 正则表达式模板
                    if re.search(template, text, re.MULTILINE):
                        score += 0.2

        # 检查结构化特征
        # 编号列表
        if re.search(r'^\d+\.', text, re.MULTILINE):
            score += 0.2

        # 项目符号
        if re.search(r'^[\*\-\•]', text, re.MULTILINE):
            score += 0.1

        # 分步骤格式
        if re.search(r'step \d|first.*then|next', text_lower):
            score += 0.2

        # 多段落（结构化回答的特征）
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 2:
            score += 0.1

        return min(score, 1.0)

    def match_qa_structure(self, text: str) -> float:
        """
        匹配问答结构

        返回: 匹配分数 (0-1)
        """
        score = 0.0

        # 检查是否有明显的问答分隔
        qa_indicators = [
            "answer:", "solution:", "response:",
            "q:", "a:", "question:",
        ]

        text_lower = text.lower()
        for indicator in qa_indicators:
            if indicator in text_lower:
                score += 0.3
                break

        # 检查是否有问题后跟答案的模式
        if "?" in text:
            # 问号后面有实质性内容
            parts = text.split("?")
            if len(parts) > 1 and len(parts[1].strip()) > 50:
                score += 0.4

        # 检查对话标记
        if re.search(r'\n(user|human|person):', text_lower) and \
                re.search(r'\n(assistant|ai|bot|response):', text_lower):
            score += 0.3

        return min(score, 1.0)

    def match_domain_keywords(self, text: str) -> float:
        """
        匹配领域关键词

        返回: 匹配分数 (0-1)
        """
        if "domain_keywords" not in self.features:
            return 0.0

        text_lower = text.lower()
        domains = self.features["domain_keywords"]

        max_score = 0.0
        for domain, keywords in domains.items():
            if not keywords:
                continue

            # 计算该领域的匹配度
            matched = sum(1 for kw in keywords if kw in text_lower)
            domain_score = min(matched / max(len(keywords), 1), 1.0)
            max_score = max(max_score, domain_score)

        return max_score

    def calculate_match_score(self, text: str) -> Tuple[float, Dict[str, float]]:
        """
        计算总体匹配分数

        返回: (总分, 各项得分)
        """
        # 添加所有匹配器
        self.matchers = []
        self.add_matcher(self.match_instruction_format, "instruction_format", 1.0)
        self.add_matcher(self.match_response_format, "response_format", 0.8)
        self.add_matcher(self.match_qa_structure, "qa_structure", 0.9)
        self.add_matcher(self.match_domain_keywords, "domain_keywords", 0.6)

        scores = {}
        weighted_sum = 0.0
        weight_total = 0.0

        for func, name, weight in self.matchers:
            score = func(text)
            scores[name] = score
            weighted_sum += score * weight
            weight_total += weight

        total_score = weighted_sum / weight_total if weight_total > 0 else 0

        return total_score, scores

    def match_pretraining_data(self, data: List[Dict], threshold: float = 0.3) -> List[Dict]:
        """
        匹配预训练数据

        Args:
            data: 预训练数据
            threshold: 匹配阈值

        返回: 匹配的数据和分数
        """
        matched_data = []

        for item in tqdm(data, desc="匹配预训练数据"):
            text = item.get("text", "")
            if not text:
                continue

            total_score, component_scores = self.calculate_match_score(text)

            if total_score >= threshold:
                item["match_score"] = total_score
                item["match_details"] = component_scores
                matched_data.append(item)

        return matched_data


def main():
    parser = argparse.ArgumentParser(description="匹配预训练数据")
    parser.add_argument("--features_file", type=str,
                        default="/root/autodl-tmp/ift_memorization/data/instruction_test_data/features/instruction_features.pkl",
                        help="指令特征文件")
    parser.add_argument("--pretraining_dir", type=str,
                        default="/root/autodl-tmp/ift_memorization/data/pretraining_test_data/mem_test",
                        help="预处理后的预训练数据目录")
    parser.add_argument("--output_dir", type=str,
                        default="/root/autodl-tmp/ift_memorization/data/pretrain_instruction_matched_data",
                        help="输出目录")
    parser.add_argument("--threshold", type=float,
                        default=0.3,
                        help="匹配阈值")
    parser.add_argument("--max_samples", type=int,
                        default=None,
                        help="每个数据集的最大处理样本数")

    args = parser.parse_args()

    # 初始化匹配器
    matcher = PretrainingMatcher(args.features_file)
    os.makedirs(args.output_dir, exist_ok=True)

    # 处理每个预训练数据集
    datasets = ["stackexchange", "wiki", "dclm"]
    suffix_map = {
        "stackexchange": "instruction",
        "wiki": "fact",
        "dclm": "privacy"
    }

    for dataset_name in datasets:
        print(f"\n处理 {dataset_name} 数据集...")

        input_file = os.path.join(args.pretraining_dir, f"{dataset_name}_{suffix_map.get(dataset_name, '')}.jsonl")
        if not os.path.exists(input_file):
            print(f"跳过不存在的文件: {input_file}")
            continue

        # 加载数据
        data = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
                if args.max_samples and len(data) >= args.max_samples:
                    break

        print(f"加载 {len(data)} 条数据")

        # 执行匹配
        matched_data = matcher.match_pretraining_data(data, args.threshold)

        print(f"匹配到 {len(matched_data)} 条数据 (阈值={args.threshold})")

        # 统计分数分布
        if matched_data:
            scores = [item["match_score"] for item in matched_data]
            print(f"分数分布: 平均={np.mean(scores):.3f}, 中位数={np.median(scores):.3f}, "
                  f"最小={np.min(scores):.3f}, 最大={np.max(scores):.3f}")

        # 保存匹配结果
        output_file = os.path.join(args.output_dir, f"{dataset_name}_matched.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for item in matched_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"保存到: {output_file}")

        # 保存高分样本用于检查
        high_score_samples = sorted(matched_data, key=lambda x: x["match_score"], reverse=True)[:10]
        sample_file = os.path.join(args.output_dir, f"{dataset_name}_high_score_samples.json")
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(high_score_samples, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()