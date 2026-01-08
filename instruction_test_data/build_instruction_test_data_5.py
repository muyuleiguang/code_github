"""
Build a dataset for testing instruction-related memorization
"""
import json
import os
import random
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm


class InstructionTestBuilder:
    def __init__(self, seed: int = 42):
        """
        Initialize the test data builder

        Args:
            seed: Random seed
        """
        random.seed(seed)
        self.builders = []

    def add_builder(self, func, name: str):
        """Add a builder function"""
        self.builders.append((func, name))

    def build_instruction_memorization_test(self, item: Dict) -> List[Dict]:
        """
        Build instruction memorization tests

        Test whether the model memorizes pretraining data with similar instruction formats
        """
        tests = []
        text = item["text"]
        match_score = item.get("match_score", 0)
        match_details = item.get("match_details", {})

        # Build different tests depending on the match type

        # 1. If it is in QA format, build a QA test
        if match_details.get("qa_structure", 0) > 0.5:
            # Try to split question and answer
            if "?" in text:
                parts = text.split("?", 1)
                if len(parts) == 2 and len(parts[1].strip()) > 50:
                    tests.append({
                        "test_type": "qa_completion",
                        "prompt": parts[0] + "?",
                        "generation": parts[1].strip(),
                        "match_score": match_score
                    })

        # 2. Instruction-format continuation test
        if match_details.get("instruction_format", 0) > 0.5:
            # Use the first 30% as the prompt
            split_point = len(text) // 3
            tests.append({
                "test_type": "instruction_continuation",
                "prompt": text[:split_point],
                "generation": text[split_point:],
                "match_score": match_score
            })

        # 3. Response-format test
        if match_details.get("response_format", 0) > 0.5:
            # If the text contains a list or steps, test format memorization
            if "\n1." in text or "\nStep" in text:
                # Provide the beginning and predict the formatted content
                lines = text.split("\n")
                if len(lines) > 3:
                    tests.append({
                        "test_type": "format_completion",
                        "prompt": "\n".join(lines[:2]),
                        "expected_format": "\n".join(lines[2:]),
                        "has_structure": True,
                        "match_score": match_score
                    })

        return tests

    def build_control_group_test(self, item: Dict) -> List[Dict]:
        """
        Build control-group tests

        Use low match-score data as controls
        """
        tests = []
        text = item["text"]
        match_score = item.get("match_score", 0)

        # Only use low-score data as controls
        if match_score < 0.2:
            # Standard prefix-continuation tests
            for ratio in [0.25, 0.5, 0.75]:
                split_point = int(len(text) * ratio)
                if split_point > 50 and split_point < len(text) - 50:
                    tests.append({
                        "test_type": "control_continuation",
                        "prompt": text[:split_point],
                        "generation": text[split_point:],
                        "prefix_ratio": ratio,
                        "match_score": match_score,
                        "is_control": True
                    })

        return tests

    def build_gradient_test(self, matched_data: List[Dict]) -> List[Dict]:
        """
        Build gradient tests

        Select data with different match scores to test the relationship
        between memorization and instruction similarity
        """
        # Group by score
        score_bins = {
            "very_low": [],  # 0-0.2
            "low": [],  # 0.2-0.4
            "medium": [],  # 0.4-0.6
            "high": [],  # 0.6-0.8
            "very_high": []  # 0.8-1.0
        }

        for item in matched_data:
            score = item.get("match_score", 0)
            if score < 0.2:
                score_bins["very_low"].append(item)
            elif score < 0.4:
                score_bins["low"].append(item)
            elif score < 0.6:
                score_bins["medium"].append(item)
            elif score < 0.8:
                score_bins["high"].append(item)
            else:
                score_bins["very_high"].append(item)

        # Sample from each bin
        gradient_tests = []
        samples_per_bin = 100

        for bin_name, bin_data in score_bins.items():
            if not bin_data:
                continue

            sampled = random.sample(bin_data, min(samples_per_bin, len(bin_data)))

            for item in sampled:
                text = item["text"]
                score = item["match_score"]

                # Standard test
                split_point = len(text) // 2
                gradient_tests.append({
                    "test_type": "gradient_test",
                    "score_bin": bin_name,
                    "match_score": score,
                    "prompt": text[:split_point],
                    "generation": text[split_point:],
                    "text_length": len(text)
                })

        return gradient_tests

    def build_all_tests(self, matched_data: List[Dict], control_data: List[Dict]) -> Dict:
        """
        Build all tests

        Args:
            matched_data: Matched data (high-score)
            control_data: Control data (low-score or unmatched)
        """
        all_tests = {
            "instruction_tests": [],
            "control_tests": [],
            "gradient_tests": []
        }

        # Build instruction-related tests
        for item in tqdm(matched_data[:1000], desc="构建指令测试"):
            tests = self.build_instruction_memorization_test(item)
            all_tests["instruction_tests"].extend(tests)

        # Build control-group tests
        for item in tqdm(control_data[:500], desc="构建对照测试"):
            tests = self.build_control_group_test(item)
            all_tests["control_tests"].extend(tests)

        # Build gradient tests
        all_data = matched_data + control_data
        gradient_tests = self.build_gradient_test(all_data)
        all_tests["gradient_tests"] = gradient_tests

        return all_tests


def main():
    parser = argparse.ArgumentParser(description="构建指令相关测试数据")
    parser.add_argument("--matched_dir", type=str, default="../../data/pretrain_instruction_matched_data", help="匹配数据目录")
    parser.add_argument("--output_dir", type=str, default="../../data/instruction_test_data", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    builder = InstructionTestBuilder(seed=args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load matched data
    all_matched_data = []
    all_control_data = []

    datasets = ["stackexchange", "wiki", "dclm"]

    for dataset_name in datasets:
        matched_file = os.path.join(args.matched_dir, f"{dataset_name}_matched.jsonl")

        if not os.path.exists(matched_file):
            print(f"跳过不存在的文件: {matched_file}")
            continue

        print(f"加载 {dataset_name} 匹配数据...")

        with open(matched_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                item["source_dataset"] = dataset_name

                if item.get("match_score", 0) >= 0.5:
                    all_matched_data.append(item)
                elif item.get("match_score", 0) < 0.2:
                    all_control_data.append(item)

    print(f"加载 {len(all_matched_data)} 条高匹配数据")
    print(f"加载 {len(all_control_data)} 条对照数据")

    # Build test data
    test_data = builder.build_all_tests(all_matched_data, all_control_data)

    # Statistics
    print("\n测试数据统计:")
    for test_type, tests in test_data.items():
        print(f"  {test_type}: {len(tests)} 条")

        if tests:
            # Count subtypes
            subtypes = {}
            for test in tests:
                subtype = test.get("test_type", "unknown")
                subtypes[subtype] = subtypes.get(subtype, 0) + 1

            for subtype, count in subtypes.items():
                print(f"    - {subtype}: {count}")

    # Save test data
    for test_type, tests in test_data.items():
        if not tests:
            continue

        output_file = os.path.join(args.output_dir, f"{test_type}.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for test in tests:
                f.write(json.dumps(test, ensure_ascii=False) + "\n")

        print(f"保存 {test_type} 到: {output_file}")

    # Save the combined test set
    all_tests = []
    for tests in test_data.values():
        all_tests.extend(tests)

    random.shuffle(all_tests)

    combined_file = os.path.join(args.output_dir, "combined_instruction_tests.jsonl")
    with open(combined_file, "w", encoding="utf-8") as f:
        for test in all_tests:
            f.write(json.dumps(test, ensure_ascii=False) + "\n")

    print(f"\n保存合并测试集到: {combined_file}")
    print(f"总测试样本数: {len(all_tests)}")


if __name__ == "__main__":
    main()
