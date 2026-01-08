"""
Extract features from instruction data for matching pretraining data
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
        """Initialize the feature extractor"""
        self.extractors = []  # Store extractor functions
        self.features = {}  # Store extracted features
        self.skipped_samples = 0  # Count skipped samples

    def add_extractor(self, func, name: str):
        """Add a feature extraction function"""
        self.extractors.append((func, name))

    def extract_instruction_verbs(self, data: List[Dict]) -> Set[str]:
        """
        Extract instruction verbs

        Why extract:
        - These verbs are core features of instructions
        - Used to identify instruction-format text in pretraining data
        """
        verb_counter = Counter()
        skipped = 0

        # Predefined list of instruction verbs
        instruction_verbs = [
            "write", "create", "explain", "describe", "analyze", "summarize",
            "translate", "generate", "provide", "list", "give", "make",
            "help", "show", "tell", "find", "identify", "compare", "evaluate",
            "discuss", "define", "demonstrate", "develop", "design", "solve"
        ]

        for item in tqdm(data, desc="提取指令动词"):
            # Check data integrity
            if not item.get("messages") or len(item["messages"]) < 1:
                skipped += 1
                continue

            instruction = item["messages"][0]['content'].lower()

            # Search for verbs
            for verb in instruction_verbs:
                if re.search(r'\b' + verb + r'\b', instruction):
                    verb_counter[verb] += 1

        # Select frequent verbs (count > 1% of total)
        threshold = len(data) * 0.01
        selected_verbs = {verb for verb, count in verb_counter.items() if count > threshold}

        print(f"提取到 {len(selected_verbs)} 个高频指令动词")
        if skipped > 0:
            print(f"跳过 {skipped} 个数据不完整的样本")

        return selected_verbs

    def extract_question_patterns(self, data: List[Dict]) -> List[str]:
        """
        Extract question patterns

        Why extract:
        - Questions are an important form of instructions
        - Helps identify QA-format pretraining data
        """
        question_words = ["what", "why", "how", "when", "where", "who", "which", "whom", "whose"]
        question_phrases = []
        phrase_counter = Counter()
        skipped = 0

        for item in tqdm(data, desc="提取疑问模式"):
            # Check data integrity
            if not item.get("messages") or len(item["messages"]) < 1:
                skipped += 1
                continue

            instruction = item["messages"][0]['content'].lower()

            # Extract phrases starting with question words
            for q_word in question_words:
                pattern = r'\b' + q_word + r'\s+\w+(?:\s+\w+)?'
                matches = re.findall(pattern, instruction)
                for match in matches:
                    phrase_counter[match] += 1

        # Select frequent patterns
        threshold = len(data) * 0.005  # 0.5%
        question_phrases = [phrase for phrase, count in phrase_counter.items() if count > threshold]

        print(f"提取到 {len(question_phrases)} 个疑问模式")
        if skipped > 0:
            print(f"跳过 {skipped} 个数据不完整的样本")

        return question_phrases

    def extract_response_templates(self, data: List[Dict]) -> List[str]:
        """
        Extract response templates

        Why extract:
        - Identify standard response formats
        - Used to filter pretraining data with similar formatting
        """
        template_patterns = []
        template_counter = Counter()
        skipped = 0

        # Predefined template starters
        template_starters = [
            "here is", "here's", "here are",
            "the answer is", "the solution is",
            "to answer", "to solve",
            "step 1:", "first,", "let me",
            "i can help", "i'll help", "i would",
            "yes,", "no,", "certainly",
        ]

        for item in tqdm(data, desc="提取回答模板"):
            # Check data integrity - need at least 2 messages
            if not item.get("messages") or len(item["messages"]) < 2:
                skipped += 1
                continue

            # Check whether the second message has content
            if not item["messages"][1].get('content'):
                skipped += 1
                continue

            response = item["messages"][1]['content'].lower()[:100]  # Only look at the first 100 characters

            for starter in template_starters:
                if response.startswith(starter):
                    template_counter[starter] += 1

        # Select frequent templates
        threshold = len(data) * 0.01
        template_patterns = [template for template, count in template_counter.items() if count > threshold]

        # Add structural template markers
        structural_patterns = [
            r'^\d+\.',  # Numbered list
            r'^[\*\-\•]',  # Bullet points
            r'^step \d+:',  # Step format
            r'```',  # Code block
        ]

        template_patterns.extend(structural_patterns)

        print(f"提取到 {len(template_patterns)} 个回答模板")
        if skipped > 0:
            print(f"跳过 {skipped} 个数据不完整的样本")

        return template_patterns

    def extract_instruction_prefixes(self, data: List[Dict]) -> List[str]:
        """
        Extract instruction prefixes

        Why extract:
        - Instructions often have fixed opening patterns
        - Helps quickly identify instruction-format text
        """
        prefix_counter = Counter()
        skipped = 0

        for item in tqdm(data, desc="提取指令前缀"):
            # Check data integrity
            if not item.get("messages") or len(item["messages"]) < 1:
                skipped += 1
                continue

            if not item["messages"][0].get('content'):
                skipped += 1
                continue

            instruction = item["messages"][0]['content']

            # Extract first 3 words as a prefix
            words = instruction.split()[:3]
            if words:
                prefix = " ".join(words).lower()
                prefix_counter[prefix] += 1

        # Select frequent prefixes
        threshold = len(data) * 0.005
        prefixes = [prefix for prefix, count in prefix_counter.items() if count > threshold]

        print(f"提取到 {len(prefixes)} 个指令前缀")
        if skipped > 0:
            print(f"跳过 {skipped} 个数据不完整的样本")

        return prefixes

    def extract_domain_keywords(self, data: List[Dict]) -> Dict[str, Set[str]]:
        """
        Extract domain keywords

        Why extract:
        - Understand the knowledge domains covered by instructions
        - Helps categorize and filter relevant pretraining data
        """
        domains = {
            "code": set(),
            "math": set(),
            "science": set(),
            "writing": set(),
            "general": set()
        }

        # Domain indicator terms
        domain_indicators = {
            "code": ["function", "variable", "loop", "array", "class", "method", "algorithm", "debug"],
            "math": ["equation", "calculate", "solve", "formula", "theorem", "proof", "derivative"],
            "science": ["hypothesis", "experiment", "theory", "research", "molecule", "physics", "biology"],
            "writing": ["essay", "paragraph", "story", "narrative", "character", "plot", "introduction"],
        }

        # Count keyword occurrences per domain
        for domain, keywords in domain_indicators.items():
            keyword_counter = Counter()
            skipped = 0

            for item in tqdm(data, desc=f"提取{domain}领域关键词"):
                # Check data integrity
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

            # Select frequent keywords
            threshold = len(data) * 0.01
            domains[domain] = {kw for kw, count in keyword_counter.items() if count > threshold}

            if skipped > 0:
                print(f"  跳过 {skipped} 个数据不完整的样本")

        # General keywords (union across domains)
        all_keywords = set()
        for kw_set in domains.values():
            all_keywords.update(kw_set)

        domains["general"] = all_keywords

        print(f"提取领域关键词: {sum(len(v) for v in domains.values())} 个")

        return domains

    def extract_all_features(self, data: List[Dict]) -> Dict:
        """Extract all features"""

        # Add all extractors
        self.add_extractor(self.extract_instruction_verbs, "instruction_verbs")
        self.add_extractor(self.extract_question_patterns, "question_patterns")
        self.add_extractor(self.extract_response_templates, "response_templates")
        self.add_extractor(self.extract_instruction_prefixes, "instruction_prefixes")
        self.add_extractor(self.extract_domain_keywords, "domain_keywords")

        # Run extraction
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

    # Load data
    print(f"加载数据: {args.input_file}")
    data = []

    # Count total lines first for the progress bar
    total_lines = 0
    with open(args.input_file, "r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1

    print(f"文件总行数: {total_lines}")

    # Load data with a progress bar
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="加载数据"):
            data.append(json.loads(line))
            if args.max_samples and len(data) >= args.max_samples:
                break

    print(f"加载 {len(data)} 条数据")

    # Extract features
    extractor = InstructionFeatureExtractor()
    features = extractor.extract_all_features(data)

    # Save features
    os.makedirs(args.output_dir, exist_ok=True)

    # Save as JSON (for inspection)
    json_path = os.path.join(args.output_dir, "instruction_features.json")

    # Convert sets to lists for JSON serialization
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

    # Save as pickle (for downstream use)
    pickle_path = os.path.join(args.output_dir, "instruction_features.pkl")
    print("保存pickle格式特征文件...")
    with open(pickle_path, "wb") as f:
        pickle.dump(features, f)

    print(f"特征保存到: {json_path} 和 {pickle_path}")

    # Print statistics
    print("\n=== 特征统计 ===")
    for name, feature in features.items():
        if isinstance(feature, (list, set)):
            print(f"{name}: {len(feature)} 个")
        elif isinstance(feature, dict):
            print(f"{name}: {sum(len(v) if isinstance(v, (list, set)) else 1 for v in feature.values())} 个")


if __name__ == "__main__":
    main()
