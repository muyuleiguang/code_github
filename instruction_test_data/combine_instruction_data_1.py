#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指令微调数据集交集提取、过滤和特征提取工具
用于提取两个指令微调数据集的交集，并进行语言过滤、长度控制、任务分类和特征提取
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
    检测文本语言（简单的基于字符的方法）

    Args:
        text: 输入文本

    Returns:
        语言代码：'en' 表示英文，'other' 表示其他语言
    """
    # 统计英文字符比例
    if not text or len(text.strip()) == 0:
        return 'other'

    # 移除空白字符
    text_no_space = text.replace(' ', '').replace('\n', '').replace('\t', '')
    if len(text_no_space) == 0:
        return 'other'

    # 统计ASCII字符（主要是英文）的比例
    ascii_count = sum(1 for c in text if ord(c) < 128)
    ascii_ratio = ascii_count / len(text)

    # 如果ASCII字符超过80%，认为是英文
    if ascii_ratio > 0.8:
        return 'en'
    else:
        return 'other'


def simple_tokenize(text: str) -> List[str]:
    """
    简单的分词函数，按空格和标点分割

    Args:
        text: 输入文本

    Returns:
        token列表
    """
    # 简单的分词：按空格分割
    return text.split()


def count_tokens(text: str) -> int:
    """
    统计文本的token数量

    Args:
        text: 输入文本

    Returns:
        token数量
    """
    return len(simple_tokenize(text))


def extract_instruction_response(item: Dict) -> Tuple[str, str]:
    """
    从数据项中提取instruction和response

    Args:
        item: 数据项字典

    Returns:
        (instruction, response) 元组
    """
    instruction = ""
    response = ""

    if 'messages' in item:
        # 处理messages格式
        for msg in item['messages']:
            if msg['role'] == 'user':
                instruction += msg['content'] + " "
            elif msg['role'] == 'assistant':
                response += msg['content'] + " "
    else:
        # 处理直接格式
        instruction = item.get("instruction", "")
        response = item.get("response", "")

    return instruction.strip(), response.strip()


def classify_task(instruction: str, response: str) -> str:
    """
    对任务进行分类

    Args:
        instruction: 指令文本
        response: 回答文本

    Returns:
        任务类别标签

    任务类别包括：
    - question_answering: 问答任务
    - generation: 生成任务
    - explanation: 解释任务
    - translation: 翻译任务
    - summarization: 总结任务
    - coding: 编程任务
    - analysis: 分析任务
    - math: 数学任务
    - classification: 分类任务
    - other: 其他任务
    """
    instruction_lower = instruction.lower()
    response_lower = response.lower()

    # 定义任务关键词
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

    # 统计每个类别的匹配分数
    scores = defaultdict(int)
    for category, keywords in task_keywords.items():
        for keyword in keywords:
            if keyword in instruction_lower:
                scores[category] += 1

    # 返回得分最高的类别
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]
    else:
        return 'other'


def extract_instruction_features(instruction: str, instruction_verbs: Set[str]) -> Dict[str, Any]:
    """
    提取指令的关键特征

    Args:
        instruction: 指令文本
        instruction_verbs: 指令动词集合

    Returns:
        特征字典，包含：
        - start_words: 开头词列表（前3个词）
        - instruction_verbs_found: 找到的指令动词列表
        - has_question: 是否包含疑问词
        - sentence_count: 句子数量
        - avg_word_length: 平均词长
    """
    tokens = simple_tokenize(instruction.lower())

    # 提取开头词（前3个）
    start_words = tokens[:3] if len(tokens) >= 3 else tokens

    # 查找指令动词
    verbs_found = [word for word in tokens if word in instruction_verbs]

    # 检测疑问词
    question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which']
    has_question = any(qw in tokens for qw in question_words) or '?' in instruction

    # 统计句子数量（简单按句号、问号、感叹号分割）
    sentence_count = len(re.split(r'[.!?]+', instruction))

    # 计算平均词长
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
    提取回答的关键特征

    Args:
        response: 回答文本

    Returns:
        特征字典，包含：
        - start_phrase: 开头短语（前5个词）
        - has_code_block: 是否包含代码块
        - has_list: 是否包含列表
        - has_steps: 是否包含步骤说明
        - paragraph_count: 段落数量
        - formality_score: 正式度评分（0-1）
    """
    tokens = simple_tokenize(response.lower())

    # 提取开头短语（前5个词）
    start_phrase = ' '.join(tokens[:5]) if len(tokens) >= 5 else ' '.join(tokens)

    # 检测代码块
    has_code_block = '```' in response or bool(re.search(r'`[^`]+`', response))

    # 检测列表（编号或项目符号）
    has_list = bool(re.search(r'^\d+\.', response, re.MULTILINE)) or \
               bool(re.search(r'^[\*\-\•]', response, re.MULTILINE))

    # 检测步骤说明
    has_steps = bool(re.search(r'step \d|first.*then|next.*step', response.lower()))

    # 统计段落数量
    paragraphs = [p for p in response.split('\n\n') if p.strip()]
    paragraph_count = len(paragraphs)

    # 简单的正式度评分：基于正式词汇的使用
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
    加载目录下所有的parquet文件并合并

    Args:
        directory: parquet文件所在目录

    Returns:
        合并后的DataFrame
    """
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")

    # 查找所有parquet文件
    parquet_files = sorted(directory_path.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"在 {directory} 中未找到parquet文件")

    print(f"在 {directory} 中找到 {len(parquet_files)} 个parquet文件")

    # 逐个读取并合并
    dataframes = []
    for file_path in tqdm(parquet_files, desc=f"加载 {directory_path.name}"):
        df = pd.read_parquet(file_path)
        dataframes.append(df)

    # 合并所有DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"合并后共有 {len(merged_df)} 条数据")

    return merged_df


def generate_row_hash(row: Dict[Any, Any], key_fields: List[str]) -> str:
    """
    为数据行生成唯一的哈希值

    Args:
        row: 数据行（字典格式）
        key_fields: 用于生成哈希的字段列表

    Returns:
        哈希值字符串
    """
    # 提取关键字段的值
    key_values = []
    for field in key_fields:
        if field in row:
            value = row[field]
            # 将值转换为字符串（处理列表、字典等复杂类型）
            if isinstance(value, (list, dict)):
                value_str = json.dumps(value, sort_keys=True, ensure_ascii=False)
            else:
                value_str = str(value)
            key_values.append(value_str)

    # 将所有关键值连接后生成哈希
    combined_str = "|||".join(key_values)
    hash_value = hashlib.md5(combined_str.encode('utf-8')).hexdigest()

    return hash_value


def find_intersection(df1: pd.DataFrame, df2: pd.DataFrame, key_fields: List[str],
                      use_hash: bool = True) -> pd.DataFrame:
    """
    找出两个数据集的交集

    Args:
        df1: 第一个数据集
        df2: 第二个数据集
        key_fields: 用于判断唯一性的字段
        use_hash: 是否使用哈希进行比较

    Returns:
        交集数据的DataFrame
    """
    print("\n开始计算交集...")

    if use_hash:
        # 使用哈希值进行比较
        print("使用哈希值方法进行比较")

        # 为第一个数据集生成哈希
        print("为数据集1生成哈希值...")
        hashes1 = set()
        for idx, row in tqdm(df1.iterrows(), total=len(df1), desc="数据集1哈希"):
            row_dict = row.to_dict()
            hash_val = generate_row_hash(row_dict, key_fields)
            hashes1.add(hash_val)

        # 为第二个数据集生成哈希并查找交集
        print("为数据集2生成哈希值并查找交集...")
        intersection_indices = []
        for idx, row in tqdm(df2.iterrows(), total=len(df2), desc="查找交集"):
            row_dict = row.to_dict()
            hash_val = generate_row_hash(row_dict, key_fields)
            if hash_val in hashes1:
                intersection_indices.append(idx)

        # 提取交集数据（从数据集2中）
        intersection_df = df2.loc[intersection_indices].reset_index(drop=True)

    else:
        # 直接使用pandas的merge进行比较（适用于简单字段）
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
    根据语言和长度过滤数据

    Args:
        df: 输入DataFrame
        instruction_min_tokens: 指令最小token数
        instruction_max_tokens: 指令最大token数
        response_min_tokens: 回答最小token数
        response_max_tokens: 回答最大token数

    Returns:
        过滤后的DataFrame，包含新增的列：
        - instruction_text: 提取的指令文本
        - response_text: 提取的回答文本
        - instruction_tokens: 指令token数
        - response_tokens: 回答token数
        - language: 语言标识
    """
    print("\n开始语言和长度过滤...")

    # 存储过滤结果
    filtered_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="过滤数据"):
        row_dict = row.to_dict()

        # 提取instruction和response
        instruction, response = extract_instruction_response(row_dict)

        # 检查是否为空
        if not instruction or not response:
            continue

        # 检查语言（必须都是英文）
        inst_lang = detect_language(instruction)
        resp_lang = detect_language(response)

        if inst_lang != 'en' or resp_lang != 'en':
            continue

        # 统计token数
        inst_tokens = count_tokens(instruction)
        resp_tokens = count_tokens(response)

        # 检查长度范围
        if not (instruction_min_tokens <= inst_tokens <= instruction_max_tokens):
            continue
        if not (response_min_tokens <= resp_tokens <= response_max_tokens):
            continue

        # 检查是否有过多的重复内容（简单检查）
        # 如果指令和回答有超过50%的重叠，可能是重复数据
        inst_words = set(instruction.lower().split())
        resp_words = set(response.lower().split())
        if inst_words and resp_words:
            overlap_ratio = len(inst_words & resp_words) / min(len(inst_words), len(resp_words))
            if overlap_ratio > 0.5:
                continue

        # 添加新字段
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
    对数据进行任务分类并提取特征

    Args:
        df: 输入DataFrame（已经包含instruction_text和response_text列）
        instruction_verbs: 指令动词集合

    Returns:
        添加了分类和特征列的DataFrame
    """
    print("\n开始任务分类和特征提取...")

    # 存储分类和特征
    task_categories = []
    instruction_features = []
    response_features = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="分类和特征提取"):
        instruction = row['instruction_text']
        response = row['response_text']

        # 任务分类
        task_category = classify_task(instruction, response)
        task_categories.append(task_category)

        # 提取指令特征
        inst_features = extract_instruction_features(instruction, instruction_verbs)
        instruction_features.append(inst_features)

        # 提取回答特征
        resp_features = extract_response_features(response)
        response_features.append(resp_features)

    # 添加到DataFrame
    df['task_category'] = task_categories
    df['instruction_features'] = instruction_features
    df['response_features'] = response_features

    # 打印分类统计
    print("\n任务类别分布：")
    category_counts = Counter(task_categories)
    for category, count in category_counts.most_common():
        print(f"  {category}: {count} ({count / len(df) * 100:.2f}%)")

    return df


def convert_to_serializable(obj):
    """
    将对象转换为JSON可序列化的格式

    Args:
        obj: 要转换的对象

    Returns:
        转换后的对象
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
    将DataFrame保存为jsonl格式

    Args:
        df: 要保存的DataFrame
        output_path: 输出文件路径
    """
    print(f"\n保存数据到: {output_path}")

    # 确保输出目录存在
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 逐行写入jsonl文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="保存到jsonl"):
            # 将行转换为字典并处理不可序列化的对象
            row_dict = row.to_dict()
            row_dict = convert_to_serializable(row_dict)

            # 写入json行
            json_line = json.dumps(row_dict, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"成功保存 {len(df)} 条数据")

    # 输出文件信息
    file_size = Path(output_path).stat().st_size
    print(f"文件大小: {file_size / (1024 * 1024):.2f} MB")


def print_sample_data(df: pd.DataFrame, n: int = 3):
    """
    打印样例数据

    Args:
        df: DataFrame
        n: 打印的样例数量
    """
    print(f"\n数据样例（前{n}条）:")
    print("=" * 80)
    for idx, row in df.head(n).iterrows():
        print(f"\n样例 {idx + 1}:")
        row_dict = row.to_dict()
        # 转换为可序列化格式
        row_dict = convert_to_serializable(row_dict)
        print(json.dumps(row_dict, ensure_ascii=False, indent=2))
        print("-" * 80)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="提取、过滤和分析两个指令微调数据集的交集")

    # 输入路径参数
    parser.add_argument("--dataset1_path", type=str,
                        default="/root/autodl-tmp/ift_memorization/data/instruction_test_data/tulu-3-sft-olmo-2-mixture",
                        help="第一个数据集的路径（包含parquet文件的目录）")
    parser.add_argument("--dataset2_path", type=str,
                        default="/root/autodl-tmp/ift_memorization/data/instruction_test_data/tulu-3-sft-olmo-2-mixture-0225",
                        help="第二个数据集的路径（包含parquet文件的目录）")

    # 输出路径参数
    parser.add_argument("--output_dir", type=str,
                        default="/root/autodl-tmp/ift_memorization/data/instruction_test_data",
                        help="输出目录路径")
    parser.add_argument("--output_filename", type=str,
                        default="olmo_instruction_tulu3_intersection.jsonl",
                        help="输出文件名")

    # 数据处理参数
    parser.add_argument("--key_fields", type=str, nargs="+", default=["messages"],
                        help="用于判断数据唯一性的字段列表")
    parser.add_argument("--hash_for_comparison", action="store_true", default=True,
                        help="是否使用哈希值进行比较（适用于复杂字段）")

    # 过滤参数
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
    """主函数"""
    # 解析参数
    args = parse_args()

    # 定义指令动词集合（合并用户提供的列表）
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

    # 合并所有指令词（转为小写并去重）
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
        # 步骤1: 加载数据集1
        print("\n步骤1: 加载数据集1")
        print("-" * 80)
        df1 = load_parquet_files(args.dataset1_path)
        print(f"数据集1列名: {list(df1.columns)}")

        # 步骤2: 加载数据集2
        print("\n步骤2: 加载数据集2")
        print("-" * 80)
        df2 = load_parquet_files(args.dataset2_path)
        print(f"数据集2列名: {list(df2.columns)}")

        # 验证关键字段是否存在
        for field in args.key_fields:
            if field not in df1.columns or field not in df2.columns:
                raise ValueError(f"关键字段 '{field}' 在某个数据集中不存在")

        # 步骤3: 计算交集
        print("\n步骤3: 计算交集")
        print("-" * 80)
        intersection_df = find_intersection(df1, df2, args.key_fields, args.hash_for_comparison)

        # 步骤4: 语言和长度过滤
        print("\n步骤4: 语言和长度过滤")
        print("-" * 80)
        filtered_df = filter_by_language_and_length(
            intersection_df,
            instruction_min_tokens=args.instruction_min_tokens,
            instruction_max_tokens=args.instruction_max_tokens,
            response_min_tokens=args.response_min_tokens,
            response_max_tokens=args.response_max_tokens
        )

        # 步骤5: 任务分类和特征提取
        print("\n步骤5: 任务分类和特征提取")
        print("-" * 80)
        final_df = classify_and_extract_features(filtered_df, instruction_verbs)

        # 步骤6: 打印样例数据
        print("\n步骤6: 数据样例")
        print("-" * 80)
        print_sample_data(final_df, n=2)

        # 步骤7: 保存结果
        print("\n步骤7: 保存结果")
        print("-" * 80)
        output_path = os.path.join(args.output_dir, args.output_filename)
        save_to_jsonl(final_df, output_path)

        # 保存统计信息
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