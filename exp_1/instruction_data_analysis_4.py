#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验1.2 - 指令微调数据集分析（修复版）
分析Tulu3指令微调数据集的结构、模式和特征，为后续构建词典和对比分析提供基础
"""

import json
import argparse
import os
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import warnings

# 忽略字体相关的警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def setup_chinese_fonts():
    """设置中文字体显示"""
    # 尝试多种中文字体
    chinese_fonts = [
        'SimHei',  # Windows 黑体
        'Heiti TC',  # macOS 黑体
        'WenQuanYi Micro Hei',  # Linux 文泉驿微米黑
        'Noto Sans CJK SC',  # Google Noto 字体
        'DejaVu Sans',  # 备用字体
    ]

    # 设置字体
    plt.rcParams['font.sans-serif'] = chinese_fonts
    plt.rcParams['axes.unicode_minus'] = False

    # 设置 seaborn 样式
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)


# 初始化中文字体设置
setup_chinese_fonts()

# 加载spacy模型，如果没有安装则使用简单的分析方法
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    print("警告: spacy模型未安装，将使用简化的语言分析方法")
    SPACY_AVAILABLE = False


def convert_numpy_types(obj):
    """
    递归转换numpy类型为Python原生类型，解决JSON序列化问题
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def load_instruction_data(data_file: str, max_samples: int = None) -> List[Dict]:
    """
    加载指令微调数据

    Args:
        data_file: 数据文件路径
        max_samples: 最大样本数，None表示加载全部

    Returns:
        data_list: 指令数据列表
    """
    if not os.path.exists(data_file):
        print(f"错误: 文件不存在 {data_file}")
        return []

    data_list = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                data_list.append(item)

                if max_samples and len(data_list) >= max_samples:
                    break
            except json.JSONDecodeError as e:
                print(f"解析JSON时出错: {e}")
                continue

    print(f"成功加载 {len(data_list)} 条指令数据")
    return data_list


def extract_instruction_response_pairs(data_list: List[Dict]) -> List[Dict]:
    """
    提取指令-回答对

    Args:
        data_list: 原始数据列表

    Returns:
        pairs: 提取的指令-回答对列表
        每个元素包含: {'instruction': str, 'response': str, 'metadata': dict}
    """
    pairs = []

    for i, item in enumerate(data_list):
        instruction = item['prompt']
        response = item['generation']
        # 确保提取到有效的指令和回答
        if instruction and response and len(instruction.strip()) > 0 and len(response.strip()) > 0:
            pairs.append({
                'instruction': instruction.strip(),
                'response': response.strip(),
            })

    print(f"成功提取 {len(pairs)} 个有效的指令-回答对")
    return pairs


def analyze_instruction_patterns(pairs: List[Dict]) -> Dict[str, Any]:
    """
    分析指令的语言模式和结构

    Args:
        pairs: 指令-回答对列表

    Returns:
        patterns: 指令模式分析结果
    """
    patterns = {
        'sentence_types': {},  # 句子类型统计
        'instruction_verbs': {},  # 指令动词统计
        'question_words': {},  # 疑问词统计
        'length_distribution': {},  # 长度分布
        'template_patterns': {},  # 模板模式
        'task_categories': {}  # 任务类别
    }

    instructions = [pair['instruction'] for pair in pairs]

    # 1. 分析句子类型
    patterns['sentence_types'] = analyze_sentence_types(instructions)

    # 2. 分析指令动词
    patterns['instruction_verbs'] = extract_instruction_verbs(instructions)

    # 3. 分析疑问词
    patterns['question_words'] = analyze_question_words(instructions)

    # 4. 长度分布分析
    patterns['length_distribution'] = analyze_length_distribution(instructions)

    # 5. 提取模板模式
    patterns['template_patterns'] = extract_template_patterns(instructions)

    # 6. 任务类别分析
    patterns['task_categories'] = categorize_tasks(instructions)

    return patterns


def analyze_sentence_types(instructions: List[str]) -> Dict[str, Any]:
    """
    分析句子类型（疑问句、祈使句、陈述句等）

    Args:
        instructions: 指令文本列表

    Returns:
        sentence_types: 句子类型统计
    """
    types = {
        'question': 0,  # 疑问句
        'imperative': 0,  # 祈使句
        'declarative': 0,  # 陈述句
        'conditional': 0  # 条件句
    }

    for instruction in instructions:
        instruction_lower = instruction.lower().strip()

        # 疑问句识别
        if (instruction.endswith('?') or
                any(instruction_lower.startswith(qw) for qw in
                    ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'can you', 'could you', 'do you',
                     'are you'])):
            types['question'] += 1

        # 祈使句识别
        elif (any(instruction_lower.startswith(verb) for verb in
                  ['write', 'create', 'generate', 'explain', 'describe', 'provide', 'give', 'make', 'list', 'tell']) or
              'please' in instruction_lower):
            types['imperative'] += 1

        # 条件句识别
        elif any(word in instruction_lower for word in ['if', 'suppose', 'assume', 'given that']):
            types['conditional'] += 1

        # 其他归为陈述句
        else:
            types['declarative'] += 1

    # 转换为比例
    total = len(instructions)
    return {
        'counts': types,
        'proportions': {k: v / total for k, v in types.items()},
        'total': total
    }


def extract_instruction_verbs(instructions: List[str]) -> Dict[str, Any]:
    """
    提取和统计指令动词

    Args:
        instructions: 指令文本列表

    Returns:
        verb_analysis: 动词分析结果
    """
    # 预定义的指令动词列表
    instruction_verbs = [
        'write', 'create', 'generate', 'explain', 'describe', 'provide', 'give',
        'make', 'list', 'tell', 'show', 'demonstrate', 'teach', 'help',
        'analyze', 'summarize', 'compare', 'evaluate', 'review', 'assess',
        'translate', 'convert', 'transform', 'modify', 'improve', 'edit',
        'solve', 'calculate', 'compute', 'find', 'identify', 'determine',
        'design', 'plan', 'organize', 'structure', 'format', 'arrange'
    ]

    verb_counts = Counter()
    verb_positions = defaultdict(list)  # 记录动词在句子中的位置

    for i, instruction in enumerate(instructions):
        instruction_lower = instruction.lower()
        words = instruction_lower.split()

        for j, word in enumerate(words):
            # 去除标点符号
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in instruction_verbs:
                verb_counts[clean_word] += 1
                verb_positions[clean_word].append({
                    'instruction_idx': i,
                    'word_position': j,
                    'relative_position': j / len(words) if len(words) > 0 else 0
                })

    # 分析动词位置分布
    position_analysis = {}
    for verb, positions in verb_positions.items():
        if positions:
            relative_pos = [p['relative_position'] for p in positions]
            position_analysis[verb] = {
                'avg_position': float(np.mean(relative_pos)),
                'std_position': float(np.std(relative_pos)),
                'first_word_count': sum(1 for p in positions if p['word_position'] == 0)
            }

    return {
        'verb_frequencies': dict(verb_counts.most_common(20)),
        'position_analysis': position_analysis,
        'total_verbs_found': int(sum(verb_counts.values())),
        'unique_verbs': len(verb_counts)
    }


def analyze_question_words(instructions: List[str]) -> Dict[str, Any]:
    """
    分析疑问词的使用

    Args:
        instructions: 指令文本列表

    Returns:
        question_analysis: 疑问词分析结果
    """
    question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'whose', 'whom']

    qword_counts = Counter()
    qword_contexts = defaultdict(list)

    for i, instruction in enumerate(instructions):
        instruction_lower = instruction.lower()
        words = instruction_lower.split()

        for j, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in question_words:
                qword_counts[clean_word] += 1

                # 提取上下文（前后各2个词）
                start = max(0, j - 2)
                end = min(len(words), j + 3)
                context = ' '.join(words[start:end])
                qword_contexts[clean_word].append(context)

    return {
        'question_word_frequencies': dict(qword_counts.most_common()),
        'contexts': {word: contexts[:5] for word, contexts in qword_contexts.items()},  # 保存前5个上下文
        'total_questions': int(sum(qword_counts.values()))
    }


def analyze_length_distribution(instructions: List[str]) -> Dict[str, Any]:
    """
    分析指令长度分布

    Args:
        instructions: 指令文本列表

    Returns:
        length_stats: 长度分布统计
    """
    char_lengths = [len(instruction) for instruction in instructions]
    word_lengths = [len(instruction.split()) for instruction in instructions]

    return {
        'character_length': {
            'mean': float(np.mean(char_lengths)),
            'std': float(np.std(char_lengths)),
            'min': int(np.min(char_lengths)),
            'max': int(np.max(char_lengths)),
            'median': float(np.median(char_lengths)),
            'percentiles': {
                '25': float(np.percentile(char_lengths, 25)),
                '75': float(np.percentile(char_lengths, 75)),
                '90': float(np.percentile(char_lengths, 90)),
                '95': float(np.percentile(char_lengths, 95))
            }
        },
        'word_length': {
            'mean': float(np.mean(word_lengths)),
            'std': float(np.std(word_lengths)),
            'min': int(np.min(word_lengths)),
            'max': int(np.max(word_lengths)),
            'median': float(np.median(word_lengths)),
            'percentiles': {
                '25': float(np.percentile(word_lengths, 25)),
                '75': float(np.percentile(word_lengths, 75)),
                '90': float(np.percentile(word_lengths, 90)),
                '95': float(np.percentile(word_lengths, 95))
            }
        },
        'raw_char_lengths': char_lengths,
        'raw_word_lengths': word_lengths
    }


def extract_template_patterns(instructions: List[str]) -> Dict[str, Any]:
    """
    提取常见的指令模板模式

    Args:
        instructions: 指令文本列表

    Returns:
        templates: 模板模式分析
    """
    # 定义常见模板模式
    template_patterns = {
        'please_verb': r'^please\s+\w+',
        'can_you': r'^can you\s+\w+',
        'could_you': r'^could you\s+\w+',
        'how_to': r'^how\s+to\s+\w+',
        'what_is': r'^what\s+is\s+',
        'explain_how': r'^explain\s+how\s+',
        'write_a': r'^write\s+a\s+\w+',
        'create_a': r'^create\s+a\s+\w+',
        'provide_me': r'provide\s+me\s+with',
        'i_need': r'^i\s+need\s+',
        'help_me': r'^help\s+me\s+'
    }

    pattern_counts = {}
    pattern_examples = {}

    for pattern_name, pattern in template_patterns.items():
        matches = []
        for instruction in instructions:
            if re.search(pattern, instruction.lower()):
                matches.append(instruction)

        pattern_counts[pattern_name] = len(matches)
        pattern_examples[pattern_name] = matches[:5]  # 保存前5个例子

    # 找出最常见的开头词
    first_words = []
    for instruction in instructions:
        words = instruction.lower().split()
        if words:
            first_words.append(words[0])

    return {
        'template_frequencies': pattern_counts,
        'template_examples': pattern_examples,
        'common_first_words': dict(Counter(first_words).most_common(20)),
        'total_instructions': len(instructions)
    }


def categorize_tasks(instructions: List[str]) -> Dict[str, Any]:
    """
    根据指令内容对任务进行分类

    Args:
        instructions: 指令文本列表

    Returns:
        categories: 任务分类结果
    """
    # 定义任务类别关键词
    task_keywords = {
        'writing': ['write', 'compose', 'draft', 'author', 'essay', 'article', 'story', 'letter'],
        'explanation': ['explain', 'describe', 'clarify', 'elaborate', 'detail'],
        'analysis': ['analyze', 'examine', 'evaluate', 'assess', 'review', 'critique'],
        'summary': ['summarize', 'summarise', 'sum up', 'brief', 'overview'],
        'translation': ['translate', 'convert', 'interpret'],
        'generation': ['generate', 'create', 'produce', 'make', 'build'],
        'comparison': ['compare', 'contrast', 'difference', 'similarity'],
        'problem_solving': ['solve', 'calculate', 'compute', 'find solution'],
        'information': ['what is', 'who is', 'when did', 'where is', 'information about'],
        'instruction': ['how to', 'steps to', 'guide', 'tutorial', 'instructions']
    }

    category_counts = defaultdict(int)
    category_examples = defaultdict(list)

    for instruction in instructions:
        instruction_lower = instruction.lower()
        categorized = False

        for category, keywords in task_keywords.items():
            for keyword in keywords:
                if keyword in instruction_lower:
                    category_counts[category] += 1
                    if len(category_examples[category]) < 3:  # 保存前3个例子
                        category_examples[category].append(instruction)
                    categorized = True
                    break
            if categorized:
                break

        if not categorized:
            category_counts['other'] += 1
            if len(category_examples['other']) < 3:
                category_examples['other'].append(instruction)

    return {
        'category_counts': dict(category_counts),
        'category_proportions': {k: v / len(instructions) for k, v in category_counts.items()},
        'category_examples': dict(category_examples),
        'total_categorized': int(sum(category_counts.values()))
    }


def analyze_response_patterns(pairs: List[Dict]) -> Dict[str, Any]:
    """
    分析回答的语言模式和结构

    Args:
        pairs: 指令-回答对列表

    Returns:
        response_patterns: 回答模式分析结果
    """
    responses = [pair['response'] for pair in pairs]

    patterns = {
        'opening_phrases': {},  # 开场白短语
        'structural_markers': {},  # 结构化标记
        'closing_phrases': {},  # 结尾短语
        'length_distribution': {},  # 长度分布
        'formality_indicators': {}  # 正式性指标
    }

    # 1. 分析开场白
    patterns['opening_phrases'] = analyze_opening_phrases(responses)

    # 2. 分析结构化标记
    patterns['structural_markers'] = analyze_structural_markers(responses)

    # 3. 分析结尾短语
    patterns['closing_phrases'] = analyze_closing_phrases(responses)

    # 4. 长度分布
    patterns['length_distribution'] = analyze_length_distribution(responses)

    # 5. 正式性指标
    patterns['formality_indicators'] = analyze_formality_indicators(responses)

    return patterns


def analyze_opening_phrases(responses: List[str]) -> Dict[str, Any]:
    """
    分析回答的开场白短语

    Args:
        responses: 回答文本列表

    Returns:
        opening_analysis: 开场白分析结果
    """
    # 定义常见开场白模式
    opening_patterns = [
        r'^(certainly|sure|of course|absolutely)',
        r'^(here are|here is|here\'s)',
        r'^(let me|i\'ll|i will)',
        r'^(to answer|to address)',
        r'^(based on|according to)',
        r'^(the answer is|the result is)',
        r'^(in order to|to)',
        r'^(first|firstly|to begin)',
        r'^(yes|no),?\s',
        r'^(thank you|thanks)'
    ]

    pattern_counts = defaultdict(int)
    pattern_examples = defaultdict(list)

    for response in responses:
        response_lower = response.lower().strip()
        first_sentence = response_lower.split('.')[0]

        for pattern in opening_patterns:
            if re.search(pattern, first_sentence):
                pattern_name = pattern.replace('^', '').replace('(', '').replace(')', '').split('|')[0]
                pattern_counts[pattern_name] += 1
                if len(pattern_examples[pattern_name]) < 3:
                    pattern_examples[pattern_name].append(response[:100] + "...")

    # 分析第一个词
    first_words = []
    for response in responses:
        words = response.split()
        if words:
            first_words.append(words[0].lower().strip('.,!?'))

    return {
        'pattern_frequencies': dict(pattern_counts),
        'pattern_examples': dict(pattern_examples),
        'common_first_words': dict(Counter(first_words).most_common(15)),
        'total_responses': len(responses)
    }


def analyze_structural_markers(responses: List[str]) -> Dict[str, Any]:
    """
    分析结构化标记的使用

    Args:
        responses: 回答文本列表

    Returns:
        structure_analysis: 结构化标记分析结果
    """
    markers = {
        'numbered_lists': r'^\s*\d+\.',
        'bullet_points': r'^\s*[•\-\*]',
        'step_markers': r'\b(step \d+|first|second|third|next|then|finally)\b',
        'section_headers': r'^\s*[A-Z][^.!?]*:$',
        'transitional': r'\b(however|moreover|furthermore|additionally|in addition|therefore|thus|consequently)\b',
        'emphasis': r'\*\*.*?\*\*|__.*?__|`.*?`'
    }

    marker_counts = defaultdict(int)
    marker_examples = defaultdict(list)

    for response in responses:
        for marker_name, pattern in markers.items():
            matches = re.findall(pattern, response, re.MULTILINE | re.IGNORECASE)
            if matches:
                marker_counts[marker_name] += len(matches)
                if len(marker_examples[marker_name]) < 3:
                    # 提取包含标记的句子
                    sentences = response.split('\n')
                    for sentence in sentences:
                        if re.search(pattern, sentence, re.IGNORECASE):
                            marker_examples[marker_name].append(sentence.strip())
                            break

    return {
        'marker_frequencies': dict(marker_counts),
        'marker_examples': dict(marker_examples),
        'responses_with_structure': sum(1 for response in responses if any(
            re.search(pattern, response, re.MULTILINE | re.IGNORECASE) for pattern in markers.values())),
        'total_responses': len(responses)
    }


def analyze_closing_phrases(responses: List[str]) -> Dict[str, Any]:
    """
    分析结尾短语

    Args:
        responses: 回答文本列表

    Returns:
        closing_analysis: 结尾短语分析结果
    """
    closing_patterns = [
        r'(let me know|feel free to ask)',
        r'(hope this helps|hope that helps)',
        r'(if you have any questions|if you need)',
        r'(thank you|thanks)',
        r'(in conclusion|to conclude|in summary)',
        r'(best regards|best wishes|good luck)'
    ]

    pattern_counts = defaultdict(int)
    pattern_examples = defaultdict(list)

    for response in responses:
        # 只分析最后一两句
        sentences = response.split('.')
        last_part = '. '.join(sentences[-2:]).lower()

        for pattern in closing_patterns:
            if re.search(pattern, last_part):
                pattern_name = pattern.replace('(', '').replace(')', '').split('|')[0]
                pattern_counts[pattern_name] += 1
                if len(pattern_examples[pattern_name]) < 3:
                    pattern_examples[pattern_name].append(last_part[:100] + "...")

    return {
        'closing_frequencies': dict(pattern_counts),
        'closing_examples': dict(pattern_examples),
        'total_responses': len(responses)
    }


def analyze_formality_indicators(responses: List[str]) -> Dict[str, Any]:
    """
    分析正式性指标

    Args:
        responses: 回答文本列表

    Returns:
        formality_analysis: 正式性分析结果
    """
    formal_indicators = [
        'furthermore', 'therefore', 'consequently', 'subsequently', 'nevertheless',
        'accordingly', 'moreover', 'however', 'thus', 'hence'
    ]

    informal_indicators = [
        'gonna', 'wanna', 'kinda', 'sorta', 'yeah', 'okay', 'ok',
        'cool', 'awesome', 'pretty good', 'like really'
    ]

    formal_count = 0
    informal_count = 0

    for response in responses:
        response_lower = response.lower()
        for indicator in formal_indicators:
            if indicator in response_lower:
                formal_count += 1
        for indicator in informal_indicators:
            if indicator in response_lower:
                informal_count += 1

    return {
        'formal_indicator_count': formal_count,
        'informal_indicator_count': informal_count,
        'formality_ratio': formal_count / (formal_count + informal_count + 1),
        'total_responses': len(responses)
    }


def visualize_instruction_analysis(instruction_patterns: Dict[str, Any],
                                   response_patterns: Dict[str, Any],
                                   output_dir: str):
    """
    可视化指令分析结果

    Args:
        instruction_patterns: 指令模式分析结果
        response_patterns: 回答模式分析结果
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 重新设置字体
    setup_chinese_fonts()

    # 1. 指令分析可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 句子类型分布
    ax1 = axes[0, 0]
    sentence_types = instruction_patterns['sentence_types']['proportions']
    if sentence_types:
        # 使用英文标签避免中文显示问题
        labels_map = {
            'question': 'Question',
            'imperative': 'Imperative',
            'declarative': 'Declarative',
            'conditional': 'Conditional'
        }
        labels = [labels_map.get(k, k) for k in sentence_types.keys()]
        values = list(sentence_types.values())
        ax1.pie(values, labels=labels, autopct='%1.1f%%')
        ax1.set_title('Instruction Sentence Types', fontsize=14)

    # 指令动词频率
    ax2 = axes[0, 1]
    verb_freq = instruction_patterns['instruction_verbs']['verb_frequencies']
    if verb_freq:
        top_verbs = list(verb_freq.items())[:15]
        verbs, counts = zip(*top_verbs) if top_verbs else ([], [])
        y_pos = np.arange(len(verbs))
        ax2.barh(y_pos, counts)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(verbs)
        ax2.set_xlabel('Frequency')
        ax2.set_title('Top 15 Instruction Verbs')
        ax2.invert_yaxis()

    # 任务类别分布
    ax3 = axes[1, 0]
    task_cats = instruction_patterns['task_categories']['category_proportions']
    if task_cats:
        cats = list(task_cats.keys())
        props = list(task_cats.values())
        ax3.bar(cats, props)
        ax3.set_xlabel('Task Categories')
        ax3.set_ylabel('Proportion')
        ax3.set_title('Task Category Distribution')
        ax3.tick_params(axis='x', rotation=45)

    # 长度分布
    ax4 = axes[1, 1]
    if 'raw_word_lengths' in instruction_patterns['length_distribution']:
        word_lengths = instruction_patterns['length_distribution']['raw_word_lengths']
        ax4.hist(word_lengths, bins=30, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Word Count')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Instruction Length Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'instruction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 回答分析可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 开场白频率
    ax1 = axes[0, 0]
    opening_freq = response_patterns['opening_phrases']['common_first_words']
    if opening_freq:
        top_openings = list(opening_freq.items())[:10]
        if top_openings:
            words, counts = zip(*top_openings)
            y_pos = np.arange(len(words))
            ax1.barh(y_pos, counts)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(words)
            ax1.set_xlabel('Frequency')
            ax1.set_title('Common Opening Words')
            ax1.invert_yaxis()

    # 结构化标记
    ax2 = axes[0, 1]
    structure_freq = response_patterns['structural_markers']['marker_frequencies']
    if structure_freq:
        markers = list(structure_freq.keys())
        counts = list(structure_freq.values())
        ax2.bar(markers, counts)
        ax2.set_xlabel('Marker Types')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Structural Marker Usage')
        ax2.tick_params(axis='x', rotation=45)

    # 正式性指标
    ax3 = axes[1, 0]
    formality = response_patterns['formality_indicators']
    formal_count = formality['formal_indicator_count']
    informal_count = formality['informal_indicator_count']
    ax3.bar(['Formal', 'Informal'], [formal_count, informal_count])
    ax3.set_ylabel('Count')
    ax3.set_title('Formality Indicators')

    # 回答长度分布
    ax4 = axes[1, 1]
    if 'raw_word_lengths' in response_patterns['length_distribution']:
        word_lengths = response_patterns['length_distribution']['raw_word_lengths']
        ax4.hist(word_lengths, bins=30, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Word Count')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Response Length Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'response_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"可视化结果已保存到: {output_dir}")


def save_analysis_results(pairs: List[Dict], instruction_patterns: Dict[str, Any],
                          response_patterns: Dict[str, Any], output_dir: str):
    """
    保存分析结果

    Args:
        pairs: 指令-回答对
        instruction_patterns: 指令模式分析结果
        response_patterns: 回答模式分析结果
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存指令-回答对数据
    pairs_file = os.path.join(output_dir, 'instruction_response_pairs.json')
    with open(pairs_file, 'w', encoding='utf-8') as f:
        # 转换numpy类型
        pairs_to_save = convert_numpy_types(pairs[:1000])  # 只保存前1000个以节省空间
        json.dump(pairs_to_save, f, ensure_ascii=False, indent=2)

    # 保存指令模式分析
    instruction_file = os.path.join(output_dir, 'instruction_patterns.json')
    with open(instruction_file, 'w', encoding='utf-8') as f:
        # 移除原始长度数据以减少文件大小，并转换numpy类型
        patterns_to_save = dict(instruction_patterns)
        if 'length_distribution' in patterns_to_save:
            patterns_to_save['length_distribution'] = {
                k: v for k, v in patterns_to_save['length_distribution'].items()
                if not k.startswith('raw_')
            }
        patterns_to_save = convert_numpy_types(patterns_to_save)
        json.dump(patterns_to_save, f, ensure_ascii=False, indent=2)

    # 保存回答模式分析
    response_file = os.path.join(output_dir, 'response_patterns.json')
    with open(response_file, 'w', encoding='utf-8') as f:
        # 移除原始长度数据以减少文件大小，并转换numpy类型
        patterns_to_save = dict(response_patterns)
        if 'length_distribution' in patterns_to_save:
            patterns_to_save['length_distribution'] = {
                k: v for k, v in patterns_to_save['length_distribution'].items()
                if not k.startswith('raw_')
            }
        patterns_to_save = convert_numpy_types(patterns_to_save)
        json.dump(patterns_to_save, f, ensure_ascii=False, indent=2)

    # 生成分析报告
    report_file = os.path.join(output_dir, 'instruction_dataset_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("指令微调数据集分析报告\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"总指令-回答对数量: {len(pairs)}\n\n")

        # 指令分析摘要
        f.write("指令分析摘要:\n")
        f.write("-" * 30 + "\n")

        sentence_types = instruction_patterns['sentence_types']['proportions']
        for stype, prop in sentence_types.items():
            f.write(f"  {stype}: {prop:.1%}\n")

        f.write(f"\n最常见的指令动词:\n")
        verb_freq = instruction_patterns['instruction_verbs']['verb_frequencies']
        for verb, count in list(verb_freq.items())[:10]:
            f.write(f"  {verb}: {count}\n")

        f.write(f"\n任务类别分布:\n")
        task_cats = instruction_patterns['task_categories']['category_proportions']
        for cat, prop in sorted(task_cats.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {cat}: {prop:.1%}\n")

        # 回答分析摘要
        f.write(f"\n\n回答分析摘要:\n")
        f.write("-" * 30 + "\n")

        opening_freq = response_patterns['opening_phrases']['common_first_words']
        f.write("最常见的开场词:\n")
        for word, count in list(opening_freq.items())[:10]:
            f.write(f"  {word}: {count}\n")

        structure_freq = response_patterns['structural_markers']['marker_frequencies']
        f.write(f"\n结构化标记使用:\n")
        for marker, count in structure_freq.items():
            f.write(f"  {marker}: {count}\n")

        formality = response_patterns['formality_indicators']
        f.write(f"\n正式性分析:\n")
        f.write(f"  正式指标出现次数: {formality['formal_indicator_count']}\n")
        f.write(f"  非正式指标出现次数: {formality['informal_indicator_count']}\n")
        f.write(f"  正式性比例: {formality['formality_ratio']:.3f}\n")

    print(f"分析结果已保存到: {output_dir}")
    print(f"  - 指令-回答对: {pairs_file}")
    print(f"  - 指令模式: {instruction_file}")
    print(f"  - 回答模式: {response_file}")
    print(f"  - 分析报告: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='分析指令微调数据集的结构和模式')

    parser.add_argument('--data_file', type=str,
                        default='/root/autodl-tmp/ift_memorization/data/instruction_test_data/combined_instruction_tests.jsonl',
                        help='指令微调数据文件路径')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='最大分析样本数，None表示全部')
    parser.add_argument('--output_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/results/exp1_2',
                        help='输出目录')

    args = parser.parse_args()

    print("开始分析指令微调数据集...")
    print(f"数据文件: {args.data_file}")
    print(f"最大样本数: {args.max_samples}")
    print(f"输出目录: {args.output_dir}")

    # 加载数据
    data_list = load_instruction_data(args.data_file, args.max_samples)

    if not data_list:
        print("错误: 未加载到任何数据")
        return

    # 提取指令-回答对
    pairs = extract_instruction_response_pairs(data_list)

    if not pairs:
        print("错误: 未提取到有效的指令-回答对")
        return

    # 分析指令模式
    print("分析指令模式...")
    instruction_patterns = analyze_instruction_patterns(pairs)

    # 分析回答模式
    print("分析回答模式...")
    response_patterns = analyze_response_patterns(pairs)

    # 可视化结果
    print("生成可视化图表...")
    visualize_instruction_analysis(instruction_patterns, response_patterns, args.output_dir)

    # 保存分析结果
    print("保存分析结果...")
    save_analysis_results(pairs, instruction_patterns, response_patterns, args.output_dir)

    print("指令数据集分析完成!")


if __name__ == "__main__":
    main()