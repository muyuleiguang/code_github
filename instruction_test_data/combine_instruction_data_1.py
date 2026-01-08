#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instruction fine-tuning dataset intersection extraction, filtering, and feature extraction tool
Used to extract the intersection of two instruction fine-tuning datasets, and perform language filtering,
length constraints, task categorization, and feature extraction
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
import pandas as pd
from tqdm import tqdm
import hashlib
import numpy as np
from collections import Counter, defaultdict


def detect_language(text: str) -> str:
    """
    Detect the language of the text (a simple character-based approach)

    Args:
        text: Input text

    Returns:
        Language code: 'en' for English, 'other' for other languages
    """
    # Compute the proportion of English characters
    if not text or len(text.strip()) == 0:
        return 'other'

    # Remove whitespace characters
    text_no_space = text.replace(' ', '').replace('\n', '').replace('\t', '')
    if len(text_no_space) == 0:
        return 'other'

    # Compute the ratio of ASCII characters (primarily English)
    ascii_count = sum(1 for c in text if ord(c) < 128)
    ascii_ratio = ascii_count / len(text)

    # If ASCII characters exceed 80%, treat as English
    if ascii_ratio > 0.8:
        return 'en'
    else:
        return 'other'


def simple_tokenize(text: str) -> List[str]:
    """
    A simple tokenization function that splits by spaces and punctuation

    Args:
        text: Input text

    Returns:
        List of tokens
    """
    # Simple tokenization: split by whitespace
    return text.split()


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in the text

    Args:
        text: Input text

    Returns:
        Number of tokens
    """
    return len(simple_tokenize(text))


def extract_instruction_response(item: Dict) -> Tuple[str, str]:
    """
    Extract instruction and response from a data item

    Args:
        item: Data item dictionary

    Returns:
        (instruction, response) tuple
    """
    instruction = ""
    response = ""

    if 'messages' in item:
        # Handle the messages format
        for msg in item['messages']:
            if msg['role'] == 'user':
                instruction += msg['content'] + " "
            elif msg['role'] == 'assistant':
                response += msg['content'] + " "
    else:
        # Handle the direct format
        instruction = item.get("instruction", "")
        response = item.get("response", "")

    return instruction.strip(), response.strip()


def classify_task(instruction: str, response: str) -> str:
    """
    Classify the task type

    Args:
        instruction: Instruction text
        response: Response text

    Returns:
        Task category label

    Task categories include:
    - question_answering: QA tasks
    - generation: Generation tasks
    - explanation: Explanation tasks
    - translation: Translation tasks
    - summarization: Summarization tasks
    - coding: Coding tasks
    - analysis: Analysis tasks
    - math: Math tasks
    - classification: Classification tasks
    - other: Other tasks
    """
    instruction_lower = instruction.lower()
    response_lower = response.lower()

    # Define task keywords
    task_keywords = {
        'question_answering': ['what', 'why', 'how', 'when', 'where', 'who', 'which', '?'],
        'generation': ['write', 'create', 'generate', 'make', 'produce', 'compose', 'draft'],
        'explanation': ['explain', 'describe', 'elaborate', 'define', 'clarify', 'interpret'],
        'translation': ['translate', 'translation', 'convert to'],
        'summarization': ['summarize', 'summary', 'brief', 'outline', 'overview'],
        'coding': ['code', 'function', 'program', 'script', 'algorithm', 'implement', 'debug'],
        'analysis': ['analyze', 'analysis', 'evaluate', 'assess', 'compare', 'examine'],
        'math': ['calculate', 'solve', 'equation', 'formula', 'compute', 'math'],
        'classification': ['classify', 'categorize', 'identify', 'determine', 'label']
    }

    # Compute matching scores for each category
    scores = defaultdict(int)
    for category, keywords in task_keywords.items():
        for keyword in keywords:
            if keyword in instruction_lower:
                scores[category] += 1

    # Return the category with the highest score
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]
    else:
        return 'other'


def extract_instruction_features(instruction: str, instruction_verbs: Set[str]) -> Dict[str, Any]:
    """
    Extract key features from an instruction

    Args:
        instruction: Instruction text
        instruction_verbs: Set of instruction verbs

    Returns:
        Feature dictionary including:
        - start_words: Leading words (first 3 words)
        - instruction_verbs_found: Instruction verbs found
        - has_question: Whether it contains question words
        - sentence_count: Number of sentences
        - avg_word_length: Average word length
    """
    tokens = simple_tokenize(instruction.lower())

    # Extract leading words (first 3)
    start_words = tokens[:3] if len(tokens) >= 3 else tokens

    # Find instruction verbs
    verbs_found = [word for word in tokens if word in instruction_verbs]

    # Detect question words
    question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which']
    has_question = any(qw in tokens for qw in question_words) or '?' in instruction

    # Count sentences (simple split by period, question mark, exclamation mark)
    sentence_count = len(re.split(r'[.!?]+', instruction))

    # Compute average word length
    avg_word_length = np.mean([len(word) for word in tokens]) if tokens else 0

    return {
        'start_words': start_words,
        'instruction_verbs_found': verbs_found,
        'has_question': has_question,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length
    }


def extract_response_features(response: str) -> Dict[str, Any]:
    """
    Extract key features from a response

    Args:
        response: Response text

    Returns:
        Feature dictionary including:
        - start_phrase: Leading phrase (first 5 words)
        - has_code_block: Whether it contains a code block
        - has_list: Whether it contains a list
        - has_steps: Whether it contains step-by-step instructions
        - paragraph_count: Number of paragraphs
        - formality_score: Formality score (0-1)
    """
    tokens = simple_tokenize(response.lower())

    # Extract leading phrase (first 5 words)
    start_phrase = ' '.join(tokens[:5]) if len(tokens) >= 5 else ' '.join(tokens)

    # Detect code blocks
    has_code_block = '```' in response or bool(re.search(r'`[^`]+`', response))

    # Detect lists (numbered or bulleted)
    has_list = bool(re.search(r'^\d+\.', response, re.MULTILINE)) or \
               bool(re.search(r'^[\*\-\•]', response, re.MULTILINE))

    # Detect step-by-step patterns
    has_steps = bool(re.search(r'step \d|first.*then|next.*step', response.lower()))

    # Count paragraphs
    paragraphs = [p for p in response.split('\n\n') if p.strip()]
    paragraph_count = len(paragraphs)

    # Simple formality score: based on the usage of formal words
    formal_words = ['however', 'therefore', 'furthermore', 'moreover', 'consequently',
                    'nevertheless', 'thus', 'hence', 'accordingly']
    formality_score = sum(1 for word in formal_words if word in response.lower()) / len(formal_words)

    return {
        'start_phrase': start_phrase,
        'has_code_block': has_code_block,
        'has_list': has_list,
        'has_steps': has_steps,
        'paragraph_count': paragraph_count,
        'formality_score': formality_score
    }


def load_parquet_files(directory: str) -> pd.DataFrame:
    """
    Load and merge all parquet files under a directory

    Args:
        directory: Directory containing parquet files

    Returns:
        Merged DataFrame
    """
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")

    # Find all parquet files
    parquet_files = sorted(directory_path.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"在 {directory} 中未找到parquet文件")

    print(f"在 {directory} 中找到 {len(parquet_files)} 个parquet文件")

    # Read and merge one by one
    dataframes = []
    for file_path in tqdm(parquet_files, desc=f"加载 {directory_path.name}"):
        df = pd.read_parquet(file_path)
        dataframes.append(df)

    # Merge all DataFrames
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"合并后共有 {len(merged_df)} 条数据")

    return merged_df


def generate_row_hash(row: Dict[Any, Any], key_fields: List[str]) -> str:
    """
    Generate a unique hash for a data row

    Args:
        row: Data row (as a dict)
        key_fields: Field list used to generate the hash

    Returns:
        Hash string
    """
    # Extract values of key fields
    key_values = []
    for field in key_fields:
        if field in row:
            value = row[field]
            # Convert to string (handle complex types like list/dict)
            if isinstance(value, (list, dict)):
                value_str = json.dumps(value, sort_keys=True, ensure_ascii=False)
            else:
                value_str = str(value)
            key_values.append(value_str)

    # Concatenate all key values and generate an MD5 hash
    combined_str = "|||".join(key_values)
    hash_value = hashlib.md5(combined_str.encode('utf-8')).hexdigest()

    return hash_value


def find_intersection(df1: pd.DataFrame, df2: pd.DataFrame, key_fields: List[str],
                      use_hash: bool = True) -> pd.DataFrame:
    """
    Find the intersection of two datasets

    Args:
        df1: First dataset
        df2: Second dataset
        key_fields: Fields used to determine uniqueness
        use_hash: Whether to use hash-based comparison

    Returns:
        DataFrame containing the intersection
    """
    print("\n开始计算交集...")

    if use_hash:
        # Compare using hash values
        print("使用哈希值方法进行比较")

        # Generate hashes for dataset 1
        print("为数据集1生成哈希值...")
        hashes1 = set()
        for idx, row in tqdm(df1.iterrows(), total=len(df1), desc="数据集1哈希"):
            row_dict = row.to_dict()
            hash_val = generate_row_hash(row_dict, key_fields)
            hashes1.add(hash_val)

        # Generate hashes for dataset 2 and find intersection
        print("为数据集2生成哈希值并查找交集...")
        intersection_indices = []
        for idx, row in tqdm(df2.iterrows(), total=len(df2), desc="查找交集"):
            row_dict = row.to_dict()
            hash_val = generate_row_hash(row_dict, key_fields)
            if hash_val in hashes1:
                intersection_indices.append(idx)

        # Extract intersection rows (from dataset 2)
        intersection_df = df2.loc[intersection_indices].reset_index(drop=True)

    else:
        # Compare using pandas merge (suitable for simple fields)
        print("使用pandas merge方法进行比较")
        intersection_df = pd.merge(
            df1, df2,
            on=key_fields,
            how='inner'
        ).drop_duplicates()

    print(f"\n交集包含 {len(intersection_df)} 条数据")
    print(f"数据集1原有: {len(df1)} 条")
    print(f"数据集2原有: {len(df2)} 条")
    print(f"交集占比: {len(intersection_df) / min(len(df1), len(df2)) * 100:.2f}%")

    return intersection_df


def filter_by_language_and_length(df: pd.DataFrame,
                                  instruction_min_tokens: int = 80,
                                  instruction_max_tokens: int = 170,
                                  response_min_tokens: int = 200,
                                  response_max_tokens: int = 300) -> pd.DataFrame:
    """
    Filter data by language and length

    Args:
        df: Input DataFrame
        instruction_min_tokens: Minimum instruction token count
        instruction_max_tokens: Maximum instruction token count
        response_min_tokens: Minimum response token count
        response_max_tokens: Maximum response token count

    Returns:
        Filtered DataFrame with additional columns:
        - instruction_text: extracted instruction text
        - response_text: extracted response text
        - instruction_tokens: instruction token count
        - response_tokens: response token count
        - language: language label
    """
    print("\n开始语言和长度过滤...")

    # Store filtered results
    filtered_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="过滤数据"):
        row_dict = row.to_dict()

        # Extract instruction and response
        instruction, response = extract_instruction_response(row_dict)

        # Check emptiness
        if not instruction or not response:
            continue

        # Language check (both must be English)
        inst_lang = detect_language(instruction)
        resp_lang = detect_language(response)

        if inst_lang != 'en' or resp_lang != 'en':
            continue

        # Count tokens
        inst_tokens = count_tokens(instruction)
        resp_tokens = count_tokens(response)

        # Check length ranges
        if not (instruction_min_tokens <= inst_tokens <= instruction_max_tokens):
            continue
        if not (response_min_tokens <= resp_tokens <= response_max_tokens):
            continue

        # Check excessive repetition (simple heuristic)
        # If instruction and response overlap by more than 50%, it may be duplicate-like data
        inst_words = set(instruction.lower().split())
        resp_words = set(response.lower().split())
        if inst_words and resp_words:
            overlap_ratio = len(inst_words & resp_words) / min(len(inst_words), len(resp_words))
            if overlap_ratio > 0.5:
                continue

        # Add new fields
        row_dict['instruction_text'] = instruction
        row_dict['response_text'] = response
        row_dict['instruction_tokens'] = inst_tokens
        row_dict['response_tokens'] = resp_tokens
        row_dict['language'] = 'en'

        filtered_data.append(row_dict)

    filtered_df = pd.DataFrame(filtered_data)

    print(f"\n过滤完成：")
    print(f"  原始数据: {len(df)} 条")
    print(f"  过滤后: {len(filtered_df)} 条")
    print(f"  保留比例: {len(filtered_df) / len(df) * 100:.2f}%")

    return filtered_df


def classify_and_extract_features(df: pd.DataFrame, instruction_verbs: Set[str]) -> pd.DataFrame:
    """
    Classify tasks and extract features

    Args:
        df: Input DataFrame (already contains instruction_text and response_text)
        instruction_verbs: Set of instruction verbs

    Returns:
        DataFrame with added classification and feature columns
    """
    print("\n开始任务分类和特征提取...")

    # Store classifications and features
    task_categories = []
    instruction_features = []
    response_features = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="分类和特征提取"):
        instruction = row['instruction_text']
        response = row['response_text']

        # Task classification
        task_category = classify_task(instruction, response)
        task_categories.append(task_category)

        # Extract instruction features
        inst_features = extract_instruction_features(instruction, instruction_verbs)
        instruction_features.append(inst_features)

        # Extract response features
        resp_features = extract_response_features(response)
        response_features.append(resp_features)

    # Add to DataFrame
    df['task_category'] = task_categories
    df['instruction_features'] = instruction_features
    df['response_features'] = response_features

    # Print classification stats
    print("\n任务类别分布：")
    category_counts = Counter(task_categories)
    for category, count in category_counts.most_common():
        print(f"  {category}: {count} ({count / len(df) * 100:.2f}%)")

    return df


def convert_to_serializable(obj):
    """
    Convert an object into a JSON-serializable format

    Args:
        obj: Object to convert

    Returns:
        Converted object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj


def save_to_jsonl(df: pd.DataFrame, output_path: str):
    """
    Save a DataFrame to JSONL format

    Args:
        df: DataFrame to save
        output_path: Output file path
    """
    print(f"\n保存数据到: {output_path}")

    # Ensure the output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write line-by-line to a jsonl file
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="保存到jsonl"):
            # Convert row to dict and handle non-serializable objects
            row_dict = row.to_dict()
            row_dict = convert_to_serializable(row_dict)

            # Write one JSON line
            json_line = json.dumps(row_dict, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"成功保存 {len(df)} 条数据")

    # Output file info
    file_size = Path(output_path).stat().st_size
    print(f"文件大小: {file_size / (1024 * 1024):.2f} MB")


def print_sample_data(df: pd.DataFrame, n: int = 3):
    """
    Print sample data

    Args:
        df: DataFrame
        n: Number of samples to print
    """
    print(f"\n数据样例（前{n}条）:")
    print("=" * 80)
    for idx, row in df.head(n).iterrows():
        print(f"\n样例 {idx + 1}:")
        row_dict = row.to_dict()
        # Convert to a serializable format
        row_dict = convert_to_serializable(row_dict)
        print(json.dumps(row_dict, ensure_ascii=False, indent=2))
        print("-" * 80)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="提取、过滤和分析两个指令微调数据集的交集")

    # Input path arguments
    parser.add_argument("--dataset1_path", type=str,
                        default="/root/autodl-tmp/ift_memorization/data/instruction_test_data/tulu-3-sft-olmo-2-mixture",
                        help="第一个数据集的路径（包含parquet文件的目录）")
    parser.add_argument("--dataset2_path", type=str,
                        default="/root/autodl-tmp/ift_memorization/data/instruction_test_data/tulu-3-sft-olmo-2-mixture-0225",
                        help="第二个数据集的路径（包含parquet文件的目录）")

    # Output path arguments
    parser.add_argument("--output_dir", type=str,
                        default="/root/autodl-tmp/ift_memorization/data/instruction_test_data",
                        help="输出目录路径")
    parser.add_argument("--output_filename", type=str,
                        default="olmo_instruction_tulu3_intersection.jsonl",
                        help="输出文件名")

    # Data processing arguments
    parser.add_argument("--key_fields", type=str, nargs="+", default=["messages"],
                        help="用于判断数据唯一性的字段列表")
    parser.add_argument("--hash_for_comparison", action="store_true", default=True,
                        help="是否使用哈希值进行比较（适用于复杂字段）")

    # Filtering arguments
    parser.add_argument("--instruction_min_tokens", type=int, default=80,
                        help="指令最小token数")
    parser.add_argument("--instruction_max_tokens", type=int, default=170,
                        help="指令最大token数")
    parser.add_argument("--response_min_tokens", type=int, default=200,
                        help="回答最小token数")
    parser.add_argument("--response_max_tokens", type=int, default=300,
                        help="回答最大token数")

    return parser.parse_args()


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()

    # Define the instruction verb set (merge user-provided lists)
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

    # Merge all instruction words (lowercase + deduplicate)
    instruction_verbs = set(word.lower() for word in instruct1 + instruct2)

    print("=" * 80)
    print("指令微调数据集交集提取、过滤和特征提取工具")
    print("=" * 80)
    print(f"\n配置信息:")
    print(f"  数据集1路径: {args.dataset1_path}")
    print(f"  数据集2路径: {args.dataset2_path}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  输出文件名: {args.output_filename}")
    print(f"  关键字段: {args.key_fields}")
    print(f"  使用哈希比较: {args.hash_for_comparison}")
    print(f"\n过滤条件:")
    print(f"  指令长度范围: {args.instruction_min_tokens}-{args.instruction_max_tokens} tokens")
    print(f"  回答长度范围: {args.response_min_tokens}-{args.response_max_tokens} tokens")
    print(f"  语言过滤: 仅保留英文")
    print(f"\n指令动词数量: {len(instruction_verbs)}")
    print()

    try:
        # Step 1: Load dataset 1
        print("\n步骤1: 加载数据集1")
        print("-" * 80)
        df1 = load_parquet_files(args.dataset1_path)
        print(f"数据集1列名: {list(df1.columns)}")

        # Step 2: Load dataset 2
        print("\n步骤2: 加载数据集2")
        print("-" * 80)
        df2 = load_parquet_files(args.dataset2_path)
        print(f"数据集2列名: {list(df2.columns)}")

        # Validate whether key fields exist
        for field in args.key_fields:
            if field not in df1.columns or field not in df2.columns:
                raise ValueError(f"关键字段 '{field}' 在某个数据集中不存在")

        # Step 3: Compute intersection
        print("\n步骤3: 计算交集")
        print("-" * 80)
        intersection_df = find_intersection(df1, df2, args.key_fields, args.hash_for_comparison)

        # Step 4: Language and length filtering
        print("\n步骤4: 语言和长度过滤")
        print("-" * 80)
        filtered_df = filter_by_language_and_length(
            intersection_df,
            instruction_min_tokens=args.instruction_min_tokens,
            instruction_max_tokens=args.instruction_max_tokens,
            response_min_tokens=args.response_min_tokens,
            response_max_tokens=args.response_max_tokens
        )

        # Step 5: Task classification and feature extraction
        print("\n步骤5: 任务分类和特征提取")
        print("-" * 80)
        final_df = classify_and_extract_features(filtered_df, instruction_verbs)

        # Step 6: Print sample data
        print("\n步骤6: 数据样例")
        print("-" * 80)
        print_sample_data(final_df, n=2)

        # Step 7: Save results
        print("\n步骤7: 保存结果")
        print("-" * 80)
        output_path = os.path.join(args.output_dir, args.output_filename)
        save_to_jsonl(final_df, output_path)

        # Save statistics
        stats_path = os.path.join(args.output_dir, args.output_filename.replace('.jsonl', '_stats.json'))
        stats = {
            'total_samples': len(final_df),
            'instruction_token_range': [args.instruction_min_tokens, args.instruction_max_tokens],
            'response_token_range': [args.response_min_tokens, args.response_max_tokens],
            'task_category_distribution': dict(Counter(final_df['task_category'])),
            'avg_instruction_tokens': float(final_df['instruction_tokens'].mean()),
            'avg_response_tokens': float(final_df['response_tokens'].mean())
        }
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"统计信息已保存到: {stats_path}")

        print("\n" + "=" * 80)
        print("处理完成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
