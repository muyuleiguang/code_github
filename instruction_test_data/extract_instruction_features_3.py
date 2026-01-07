"""
从指令数据中提取特征，用于匹配预训练数据
"""
import json
import os
import re
import argparse
from typing import List, Dict, Set, Tuple
from collections import Counter
import pickle
from tqdm import tqdm


class InstructionFeatureExtractor:
    def __init__(self):
        """初始化特征提取器"""
        self.extractors = []  # 存储提取函数
        self.features = {}  # 存储提取的特征
        self.skipped_samples = 0  # 记录跳过的样本数

    def add_extractor(self, func, name: str):
        """添加特征提取函数"""
        self.extractors.append((func, name))

    def extract_instruction_verbs(self, data: List[Dict]) -> Set[str]:
        """
        提取指令动词

        为什么提取：
        - 这些动词是指令的核心特征
        - 用于识别预训练数据中的指令格式文本
        """
        verb_counter = Counter()
        skipped = 0

        # 预定义的指令动词列表
        instruction_verbs = [
            "write", "create", "explain", "describe", "analyze", "summarize",
            "translate", "generate", "provide", "list", "give", "make",
            "help", "show", "tell", "find", "identify", "compare", "evaluate",
            "discuss", "define", "demonstrate", "develop", "design", "solve"
        ]

        for item in tqdm(data, desc="提取指令动词"):
            # 检查数据完整性
            if not item.get("messages") or len(item["messages"]) < 1:
                skipped += 1
                continue

            instruction = item["messages"][0]['content'].lower()

            # 查找动词
            for verb in instruction_verbs:
                if re.search(r'\b' + verb + r'\b', instruction):
                    verb_counter[verb] += 1

        # 选择高频动词（出现次数 > 总数的1%）
        threshold = len(data) * 0.01
        selected_verbs = {verb for verb, count in verb_counter.items() if count > threshold}

        print(f"提取到 {len(selected_verbs)} 个高频指令动词")
        if skipped > 0:
            print(f"跳过 {skipped} 个数据不完整的样本")

        return selected_verbs

    def extract_question_patterns(self, data: List[Dict]) -> List[str]:
        """
        提取疑问模式

        为什么提取：
        - 疑问句是指令的重要形式
        - 帮助识别QA格式的预训练数据
        """
        question_words = ["what", "why", "how", "when", "where", "who", "which", "whom", "whose"]
        question_phrases = []
        phrase_counter = Counter()
        skipped = 0

        for item in tqdm(data, desc="提取疑问模式"):
            # 检查数据完整性
            if not item.get("messages") or len(item["messages"]) < 1:
                skipped += 1
                continue

            instruction = item["messages"][0]['content'].lower()

            # 提取疑问词开头的短语
            for q_word in question_words:
                pattern = r'\b' + q_word + r'\s+\w+(?:\s+\w+)?'
                matches = re.findall(pattern, instruction)
                for match in matches:
                    phrase_counter[match] += 1

        # 选择高频模式
        threshold = len(data) * 0.005  # 0.5%
        question_phrases = [phrase for phrase, count in phrase_counter.items() if count > threshold]

        print(f"提取到 {len(question_phrases)} 个疑问模式")
        if skipped > 0:
            print(f"跳过 {skipped} 个数据不完整的样本")

        return question_phrases

    def extract_response_templates(self, data: List[Dict]) -> List[str]:
        """
        提取回答模板

        为什么提取：
        - 识别标准的回答格式
        - 用于筛选具有类似格式的预训练数据
        """
        template_patterns = []
        template_counter = Counter()
        skipped = 0

        # 预定义的模板开头
        template_starters = [
            "here is", "here's", "here are",
            "the answer is", "the solution is",
            "to answer", "to solve",
            "step 1:", "first,", "let me",
            "i can help", "i'll help", "i would",
            "yes,", "no,", "certainly",
        ]

        for item in tqdm(data, desc="提取回答模板"):
            # 检查数据完整性 - 需要至少2条消息
            if not item.get("messages") or len(item["messages"]) < 2:
                skipped += 1
                continue

            # 检查第二条消息是否有content
            if not item["messages"][1].get('content'):
                skipped += 1
                continue

            response = item["messages"][1]['content'].lower()[:100]  # 只看开头100字符

            for starter in template_starters:
                if response.startswith(starter):
                    template_counter[starter] += 1

        # 选择高频模板
        threshold = len(data) * 0.01
        template_patterns = [template for template, count in template_counter.items() if count > threshold]

        # 添加结构化模板标记
        structural_patterns = [
            r'^\d+\.',  # 数字列表
            r'^[\*\-\•]',  # 项目符号
            r'^step \d+:',  # 步骤格式
            r'```',  # 代码块
        ]

        template_patterns.extend(structural_patterns)

        print(f"提取到 {len(template_patterns)} 个回答模板")
        if skipped > 0:
            print(f"跳过 {skipped} 个数据不完整的样本")

        return template_patterns

    def extract_instruction_prefixes(self, data: List[Dict]) -> List[str]:
        """
        提取指令前缀

        为什么提取：
        - 指令通常有固定的开头模式
        - 帮助快速识别指令格式文本
        """
        prefix_counter = Counter()
        skipped = 0

        for item in tqdm(data, desc="提取指令前缀"):
            # 检查数据完整性
            if not item.get("messages") or len(item["messages"]) < 1:
                skipped += 1
                continue

            if not item["messages"][0].get('content'):
                skipped += 1
                continue

            instruction = item["messages"][0]['content']

            # 提取前3个词作为前缀
            words = instruction.split()[:3]
            if words:
                prefix = " ".join(words).lower()
                prefix_counter[prefix] += 1

        # 选择高频前缀
        threshold = len(data) * 0.005
        prefixes = [prefix for prefix, count in prefix_counter.items() if count > threshold]

        print(f"提取到 {len(prefixes)} 个指令前缀")
        if skipped > 0:
            print(f"跳过 {skipped} 个数据不完整的样本")

        return prefixes

    def extract_domain_keywords(self, data: List[Dict]) -> Dict[str, Set[str]]:
        """
        提取领域关键词

        为什么提取：
        - 了解指令覆盖的知识领域
        - 帮助分类和筛选相关的预训练数据
        """
        domains = {
            "code": set(),
            "math": set(),
            "science": set(),
            "writing": set(),
            "general": set()
        }

        # 领域特征词
        domain_indicators = {
            "code": ["function", "variable", "loop", "array", "class", "method", "algorithm", "debug"],
            "math": ["equation", "calculate", "solve", "formula", "theorem", "proof", "derivative"],
            "science": ["hypothesis", "experiment", "theory", "research", "molecule", "physics", "biology"],
            "writing": ["essay", "paragraph", "story", "narrative", "character", "plot", "introduction"],
        }

        # 统计每个领域的关键词出现情况
        for domain, keywords in domain_indicators.items():
            keyword_counter = Counter()
            skipped = 0

            for item in tqdm(data, desc=f"提取{domain}领域关键词"):
                # 检查数据完整性
                if not item.get("messages") or len(item["messages"]) < 2:
                    skipped += 1
                    continue

                if not item["messages"][0].get('content') or not item["messages"][1].get('content'):
                    skipped += 1
                    continue

                text = (item["messages"][0]['content'] + " " + item["messages"][1]['content']).lower()
                for keyword in keywords:
                    if keyword in text:
                        keyword_counter[keyword] += 1

            # 选择高频关键词
            threshold = len(data) * 0.01
            domains[domain] = {kw for kw, count in keyword_counter.items() if count > threshold}

            if skipped > 0:
                print(f"  跳过 {skipped} 个数据不完整的样本")

        # 通用关键词（出现在多个领域）
        all_keywords = set()
        for kw_set in domains.values():
            all_keywords.update(kw_set)

        domains["general"] = all_keywords

        print(f"提取领域关键词: {sum(len(v) for v in domains.values())} 个")

        return domains

    def extract_all_features(self, data: List[Dict]) -> Dict:
        """提取所有特征"""

        # 添加所有提取器
        self.add_extractor(self.extract_instruction_verbs, "instruction_verbs")
        self.add_extractor(self.extract_question_patterns, "question_patterns")
        self.add_extractor(self.extract_response_templates, "response_templates")
        self.add_extractor(self.extract_instruction_prefixes, "instruction_prefixes")
        self.add_extractor(self.extract_domain_keywords, "domain_keywords")

        # 执行提取
        features = {}
        for func, name in self.extractors:
            print(f"\n提取特征: {name}")
            features[name] = func(data)

        return features


def main():
    parser = argparse.ArgumentParser(description="提取指令特征")
    parser.add_argument("--input_file", type=str,
                        default="/root/autodl-tmp/ift_memorization/data/instruction_test_data/olmo_instruction_tulu3_intersection.jsonl",
                        help="输入文件路径")
    parser.add_argument("--output_dir", type=str,
                        default="/root/autodl-tmp/ift_memorization/data/instruction_test_data/features",
                        help="输出目录")
    parser.add_argument("--max_samples", type=int,
                        default=None,
                        help="最大处理样本数")

    args = parser.parse_args()

    # 加载数据
    print(f"加载数据: {args.input_file}")
    data = []

    # 先计算总行数用于进度条
    total_lines = 0
    with open(args.input_file, "r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1

    print(f"文件总行数: {total_lines}")

    # 使用进度条加载数据
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="加载数据"):
            data.append(json.loads(line))
            if args.max_samples and len(data) >= args.max_samples:
                break

    print(f"加载 {len(data)} 条数据")

    # 提取特征
    extractor = InstructionFeatureExtractor()
    features = extractor.extract_all_features(data)

    # 保存特征
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存为JSON（用于查看）
    json_path = os.path.join(args.output_dir, "instruction_features.json")

    # 转换set为list以便JSON序列化
    json_features = {}
    for key, value in features.items():
        if isinstance(value, set):
            json_features[key] = list(value)
        elif isinstance(value, dict):
            json_features[key] = {k: list(v) if isinstance(v, set) else v for k, v in value.items()}
        else:
            json_features[key] = value

    print("保存JSON格式特征文件...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_features, f, indent=2, ensure_ascii=False)

    # 保存为pickle（用于后续使用）
    pickle_path = os.path.join(args.output_dir, "instruction_features.pkl")
    print("保存pickle格式特征文件...")
    with open(pickle_path, "wb") as f:
        pickle.dump(features, f)

    print(f"特征保存到: {json_path} 和 {pickle_path}")

    # 打印统计
    print("\n=== 特征统计 ===")
    for name, feature in features.items():
        if isinstance(feature, (list, set)):
            print(f"{name}: {len(feature)} 个")
        elif isinstance(feature, dict):
            print(f"{name}: {sum(len(v) if isinstance(v, (list, set)) else 1 for v in feature.values())} 个")


if __name__ == "__main__":
    main()