#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 1.2 - Instruction-tuning dataset analysis (fixed version)
Analyze the structure, patterns, and features of the Tulu3 instruction-tuning dataset,
providing a foundation for subsequent dictionary construction and comparative analysis.
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

# Ignore font-related warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def setup_chinese_fonts():
    """Configure Chinese font rendering"""
    # Try multiple Chinese fonts
    chinese_fonts = [
        'SimHei',  # Windows HeiTi
        'Heiti TC',  # macOS HeiTi
        'WenQuanYi Micro Hei',  # Linux WenQuanYi Micro Hei
        'Noto Sans CJK SC',  # Google Noto font
        'DejaVu Sans',  # Fallback font
    ]

    # Set fonts
    plt.rcParams['font.sans-serif'] = chinese_fonts
    plt.rcParams['axes.unicode_minus'] = False

    # Set seaborn style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)


# Initialize Chinese font settings
setup_chinese_fonts()

# Load spaCy model; if unavailable, use a simplified analysis method
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    print("警告: spacy模型未安装，将使用简化的语言分析方法")
    SPACY_AVAILABLE = False


def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to native Python types to resolve JSON serialization issues
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
    Load instruction-tuning data

    Args:
        data_file: Path to the data file
        max_samples: Maximum number of samples; None means load all

    Returns:
        data_list: List of instruction data
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
    Extract instruction-response pairs

    Args:
        data_list: Original data list

    Returns:
        pairs: Extracted list of instruction-response pairs
        Each element contains: {'instruction': str, 'response': str, 'metadata': dict}
    """
    pairs = []

    for i, item in enumerate(data_list):
        instruction = item['prompt']
        response = item['generation']
        # Ensure valid instruction and response are extracted
        if instruction and response and len(instruction.strip()) > 0 and len(response.strip()) > 0:
            pairs.append({
                'instruction': instruction.strip(),
                'response': response.strip(),
            })

    print(f"成功提取 {len(pairs)} 个有效的指令-回答对")
    return pairs


def analyze_instruction_patterns(pairs: List[Dict]) -> Dict[str, Any]:
    """
    Analyze instruction language patterns and structure

    Args:
        pairs: List of instruction-response pairs

    Returns:
        patterns: Instruction pattern analysis results
    """
    patterns = {
        'sentence_types': {},  # Sentence type statistics
        'instruction_verbs': {},  # Instruction verb statistics
        'question_words': {},  # Question word statistics
        'length_distribution': {},  # Length distribution
        'template_patterns': {},  # Template patterns
        'task_categories': {}  # Task categories
    }

    instructions = [pair['instruction'] for pair in pairs]

    # 1. Analyze sentence types
    patterns['sentence_types'] = analyze_sentence_types(instructions)

    # 2. Analyze instruction verbs
    patterns['instruction_verbs'] = extract_instruction_verbs(instructions)

    # 3. Analyze question words
    patterns['question_words'] = analyze_question_words(instructions)

    # 4. Length distribution analysis
    patterns['length_distribution'] = analyze_length_distribution(instructions)

    # 5. Extract template patterns
    patterns['template_patterns'] = extract_template_patterns(instructions)

    # 6. Task category analysis
    patterns['task_categories'] = categorize_tasks(instructions)

    return patterns


def analyze_sentence_types(instructions: List[str]) -> Dict[str, Any]:
    """
    Analyze sentence types (questions, imperatives, declaratives, etc.)

    Args:
        instructions: List of instruction texts

    Returns:
        sentence_types: Sentence type statistics
    """
    types = {
        'question': 0,  # Question
        'imperative': 0,  # Imperative
        'declarative': 0,  # Declarative
        'conditional': 0  # Conditional
    }

    for instruction in instructions:
        instruction_lower = instruction.lower().strip()

        # Question detection
        if (instruction.endswith('?') or
                any(instruction_lower.startswith(qw) for qw in
                    ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'can you', 'could you', 'do you',
                     'are you'])):
            types['question'] += 1

        # Imperative detection
        elif (any(instruction_lower.startswith(verb) for verb in
                  ['write', 'create', 'generate', 'explain', 'describe', 'provide', 'give', 'make', 'list', 'tell']) or
              'please' in instruction_lower):
            types['imperative'] += 1

        # Conditional detection
        elif any(word in instruction_lower for word in ['if', 'suppose', 'assume', 'given that']):
            types['conditional'] += 1

        # Otherwise treat as declarative
        else:
            types['declarative'] += 1

    # Convert to proportions
    total = len(instructions)
    return {
        'counts': types,
        'proportions': {k: v / total for k, v in types.items()},
        'total': total
    }


def extract_instruction_verbs(instructions: List[str]) -> Dict[str, Any]:
    """
    Extract and count instruction verbs

    Args:
        instructions: List of instruction texts

    Returns:
        verb_analysis: Verb analysis results
    """
    # Predefined list of instruction verbs
    instruction_verbs = [
        'write', 'create', 'generate', 'explain', 'describe', 'provide', 'give',
        'make', 'list', 'tell', 'show', 'demonstrate', 'teach', 'help',
        'analyze', 'summarize', 'compare', 'evaluate', 'review', 'assess',
        'translate', 'convert', 'transform', 'modify', 'improve', 'edit',
        'solve', 'calculate', 'compute', 'find', 'identify', 'determine',
        'design', 'plan', 'organize', 'structure', 'format', 'arrange'
    ]

    verb_counts = Counter()
    verb_positions = defaultdict(list)  # Record verb positions within sentences

    for i, instruction in enumerate(instructions):
        instruction_lower = instruction.lower()
        words = instruction_lower.split()

        for j, word in enumerate(words):
            # Remove punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in instruction_verbs:
                verb_counts[clean_word] += 1
                verb_positions[clean_word].append({
                    'instruction_idx': i,
                    'word_position': j,
                    'relative_position': j / len(words) if len(words) > 0 else 0
                })

    # Analyze verb position distributions
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
    Analyze usage of question words

    Args:
        instructions: List of instruction texts

    Returns:
        question_analysis: Question word analysis results
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

                # Extract context (2 words before and after)
                start = max(0, j - 2)
                end = min(len(words), j + 3)
                context = ' '.join(words[start:end])
                qword_contexts[clean_word].append(context)

    return {
        'question_word_frequencies': dict(qword_counts.most_common()),
        'contexts': {word: contexts[:5] for word, contexts in qword_contexts.items()},  # Keep first 5 contexts
        'total_questions': int(sum(qword_counts.values()))
    }


def analyze_length_distribution(instructions: List[str]) -> Dict[str, Any]:
    """
    Analyze instruction length distribution

    Args:
        instructions: List of instruction texts

    Returns:
        length_stats: Length distribution statistics
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
    Extract common instruction template patterns

    Args:
        instructions: List of instruction texts

    Returns:
        templates: Template pattern analysis
    """
    # Define common template patterns
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
        pattern_examples[pattern_name] = matches[:5]  # Keep first 5 examples

    # Find the most common first words
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
    Categorize tasks based on instruction content

    Args:
        instructions: List of instruction texts

    Returns:
        categories: Task categorization results
    """
    # Define task-category keywords
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
                    if len(category_examples[category]) < 3:  # Keep first 3 examples
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
    Analyze response language patterns and structure

    Args:
        pairs: List of instruction-response pairs

    Returns:
        response_patterns: Response pattern analysis results
    """
    responses = [pair['response'] for pair in pairs]

    patterns = {
        'opening_phrases': {},  # Opening phrases
        'structural_markers': {},  # Structural markers
        'closing_phrases': {},  # Closing phrases
        'length_distribution': {},  # Length distribution
        'formality_indicators': {}  # Formality indicators
    }

    # 1. Analyze openings
    patterns['opening_phrases'] = analyze_opening_phrases(responses)

    # 2. Analyze structural markers
    patterns['structural_markers'] = analyze_structural_markers(responses)

    # 3. Analyze closing phrases
    patterns['closing_phrases'] = analyze_closing_phrases(responses)

    # 4. Length distribution
    patterns['length_distribution'] = analyze_length_distribution(responses)

    # 5. Formality indicators
    patterns['formality_indicators'] = analyze_formality_indicators(responses)

    return patterns


def analyze_opening_phrases(responses: List[str]) -> Dict[str, Any]:
    """
    Analyze response opening phrases

    Args:
        responses: List of response texts

    Returns:
        opening_analysis: Opening analysis results
    """
    # Define common opening patterns
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

    # Analyze first word
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
    Analyze usage of structural markers

    Args:
        responses: List of response texts

    Returns:
        structure_analysis: Structural marker analysis results
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
                    # Extract sentences containing the marker
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
    Analyze closing phrases

    Args:
        responses: List of response texts

    Returns:
        closing_analysis: Closing phrase analysis results
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
        # Only analyze the last one or two sentences
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
    Analyze formality indicators

    Args:
        responses: List of response texts

    Returns:
        formality_analysis: Formality analysis results
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
    Visualize instruction analysis results

    Args:
        instruction_patterns: Instruction pattern analysis results
        response_patterns: Response pattern analysis results
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Reset font configuration
    setup_chinese_fonts()

    # 1. Instruction analysis visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Sentence type distribution
    ax1 = axes[0, 0]
    sentence_types = instruction_patterns['sentence_types']['proportions']
    if sentence_types:
        # Use English labels to avoid Chinese rendering issues
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

    # Instruction verb frequency
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

    # Task category distribution
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

    # Length distribution
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

    # 2. Response analysis visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Opening frequency
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

    # Structural markers
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

    # Formality indicators
    ax3 = axes[1, 0]
    formality = response_patterns['formality_indicators']
    formal_count = formality['formal_indicator_count']
    informal_count = formality['informal_indicator_count']
    ax3.bar(['Formal', 'Informal'], [formal_count, informal_count])
    ax3.set_ylabel('Count')
    ax3.set_title('Formality Indicators')

    # Response length distribution
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
    Save analysis results

    Args:
        pairs: Instruction-response pairs
        instruction_patterns: Instruction pattern analysis results
        response_patterns: Response pattern analysis results
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save instruction-response pair data
    pairs_file = os.path.join(output_dir, 'instruction_response_pairs.json')
    with open(pairs_file, 'w', encoding='utf-8') as f:
        # Convert NumPy types
        pairs_to_save = convert_numpy_types(pairs[:1000])  # Only save the first 1000 to reduce size
        json.dump(pairs_to_save, f, ensure_ascii=False, indent=2)

    # Save instruction pattern analysis
    instruction_file = os.path.join(output_dir, 'instruction_patterns.json')
    with open(instruction_file, 'w', encoding='utf-8') as f:
        # Remove raw length fields to reduce file size, and convert NumPy types
        patterns_to_save = dict(instruction_patterns)
        if 'length_distribution' in patterns_to_save:
            patterns_to_save['length_distribution'] = {
                k: v for k, v in patterns_to_save['length_distribution'].items()
                if not k.startswith('raw_')
            }
        patterns_to_save = convert_numpy_types(patterns_to_save)
        json.dump(patterns_to_save, f, ensure_ascii=False, indent=2)

    # Save response pattern analysis
    response_file = os.path.join(output_dir, 'response_patterns.json')
    with open(response_file, 'w', encoding='utf-8') as f:
        # Remove raw length fields to reduce file size, and convert NumPy types
        patterns_to_save = dict(response_patterns)
        if 'length_distribution' in patterns_to_save:
            patterns_to_save['length_distribution'] = {
                k: v for k, v in patterns_to_save['length_distribution'].items()
                if not k.startswith('raw_')
            }
        patterns_to_save = convert_numpy_types(patterns_to_save)
        json.dump(patterns_to_save, f, ensure_ascii=False, indent=2)

    # Generate analysis report
    report_file = os.path.join(output_dir, 'instruction_dataset_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("指令微调数据集分析报告\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"总指令-回答对数量: {len(pairs)}\n\n")

        # Instruction analysis summary
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

        # Response analysis summary
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

    # Load data
    data_list = load_instruction_data(args.data_file, args.max_samples)

    if not data_list:
        print("错误: 未加载到任何数据")
        return

    # Extract instruction-response pairs
    pairs = extract_instruction_response_pairs(data_list)

    if not pairs:
        print("错误: 未提取到有效的指令-回答对")
        return

    # Analyze instruction patterns
    print("分析指令模式...")
    instruction_patterns = analyze_instruction_patterns(pairs)

    # Analyze response patterns
    print("分析回答模式...")
    response_patterns = analyze_response_patterns(pairs)

    # Visualize results
    print("生成可视化图表...")
    visualize_instruction_analysis(instruction_patterns, response_patterns, args.output_dir)

    # Save analysis results
    print("保存分析结果...")
    save_analysis_results(pairs, instruction_patterns, response_patterns, args.output_dir)

    print("指令数据集分析完成!")


if __name__ == "__main__":
    main()
