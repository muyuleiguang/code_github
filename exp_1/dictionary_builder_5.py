#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建指令词典：从Tulu3指令微调数据集中提取指令特征词典
包括：指令动词词典、指令结构模式库、回答模式词汇表
"""

import argparse
import json
import re
import os
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple
import nltk
from tqdm import tqdm
import pickle

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer


class InstructionDictionaryBuilder:
    """指令词典构建器"""

    def __init__(self,
                 predefined_verbs: List[str] = None,
                 top_k_verbs: int = 500,
                 top_k_patterns: int = 30,
                 response_prefix_len: int = 100):
        """
        初始化构建器

        参数:
            predefined_verbs: 预定义的指令动词列表
            top_k_verbs: 提取高频动词的数量，默认500
            top_k_patterns: 提取高频句式模板的数量，默认30
            response_prefix_len: 分析response开头的token数，默认100
        """
        self.lemmatizer = WordNetLemmatizer()
        self.top_k_verbs = top_k_verbs
        self.top_k_patterns = top_k_patterns
        self.response_prefix_len = response_prefix_len

        # 预定义的指令动词
        self.predefined_verbs = set(predefined_verbs) if predefined_verbs else set()

        # 存储分析结果
        self.instruction_verbs = Counter()  # 指令动词频率
        self.question_words = Counter()  # 疑问词频率
        self.instruction_patterns = Counter()  # 指令句式模式
        self.response_starters = Counter()  # 回答开头模式
        self.task_types = defaultdict(list)  # 任务类型分类

    def load_dataset(self, data_path: str) -> List[Dict]:
        """
        加载指令数据集

        输入:
            data_path: JSONL格式的数据集路径
        输出:
            数据样本列表，每个样本包含instruction和response字段
        """
        print(f"加载数据集: {data_path}")
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                data.append(sample)
        print(f"成功加载 {len(data)} 条数据")
        return data

    def extract_verbs_from_instruction(self, instruction: str) -> List[str]:
        """
        从指令中提取动词

        输入:
            instruction: 指令文本
        输出:
            动词原形列表
        处理:
            1. 分词和词性标注
            2. 提取动词(VB, VBP, VBZ等)
            3. 转换为动词原形
        """
        tokens = word_tokenize(instruction.lower())
        pos_tags = pos_tag(tokens)

        verbs = []
        for word, pos in pos_tags:
            # VB: 动词原形, VBP: 动词现在时, VBZ: 动词第三人称单数
            # VBG: 动词现在分词, VBN: 动词过去分词, VBD: 动词过去式
            if pos.startswith('VB'):
                verb_lemma = self.lemmatizer.lemmatize(word, 'v')
                verbs.append(verb_lemma)

        return verbs

    def extract_question_patterns(self, instruction: str) -> Dict[str, any]:
        """
        提取疑问句模式和结构

        输入:
            instruction: 指令文本
        输出:
            包含疑问词、句式类型等信息的字典
        """
        instruction_lower = instruction.lower().strip()

        patterns = {
            'is_question': instruction.endswith('?'),
            'question_word': None,
            'sentence_type': 'statement',  # statement, question, request
            'has_please': 'please' in instruction_lower,
            'has_could_you': 'could you' in instruction_lower or 'can you' in instruction_lower,
        }

        # 提取疑问词
        question_words = ['what', 'how', 'why', 'where', 'when', 'who', 'which', 'whose']
        for qw in question_words:
            if instruction_lower.startswith(qw):
                patterns['question_word'] = qw
                patterns['sentence_type'] = 'question'
                break

        # 检测请求句式
        request_patterns = ['please', 'could you', 'can you', 'would you', 'will you']
        for rp in request_patterns:
            if rp in instruction_lower:
                patterns['sentence_type'] = 'request'
                break

        return patterns

    def extract_instruction_templates(self, instruction: str) -> List[str]:
        """
        提取指令模板

        输入:
            instruction: 指令文本
        输出:
            匹配的模板列表
        处理:
            使用正则表达式匹配常见的指令模板
        """
        templates = []
        instruction_lower = instruction.lower()

        # 定义常见的指令模板模式
        template_patterns = [
            (r'^please\s+(\w+)\s+(?:the|a|an)\s+(\w+)', 'Please [VERB] the [NOUN]'),
            (r'^can you\s+(\w+)', 'Can you [VERB]'),
            (r'^could you\s+(\w+)', 'Could you [VERB]'),
            (r'^(\w+)\s+(?:the|a|an)\s+following', '[VERB] the following'),
            (r'^what (?:is|are)\s+(?:the\s+)?(\w+)', 'What is/are [NOUN]'),
            (r'^how (?:do|does|did|can|could)\s+(\w+)', 'How [AUX] [SUBJECT]'),
            (r'^why (?:is|are|do|does|did)\s+(\w+)', 'Why [AUX] [SUBJECT]'),
            (r'^explain\s+(?:how|why|what)\s+(\w+)', 'Explain [QW] [TOPIC]'),
            (r'^describe\s+(?:the|a|an)?\s*(\w+)', 'Describe [NOUN]'),
            (r'^list\s+(?:the|all)?\s*(\w+)', 'List [NOUN]'),
            (r'^provide\s+(?:a|an|the)?\s*(\w+)', 'Provide [NOUN]'),
            (r'^write\s+(?:a|an|the)?\s*(\w+)', 'Write [NOUN]'),
        ]

        for pattern, template_name in template_patterns:
            if re.search(pattern, instruction_lower):
                templates.append(template_name)

        return templates

    def extract_response_starters(self, response: str, max_words: int = 10) -> List[str]:
        """
        提取回答的开头模式

        输入:
            response: 回答文本
            max_words: 提取开头的最大词数
        输出:
            开头模式列表
        """
        starters = []
        response_lower = response.lower().strip()

        # 获取开头的几个词
        words = response_lower.split()[:max_words]
        response_start = ' '.join(words)

        # 检测常见的开头模式
        starter_patterns = [
            (r'^(certainly|sure|absolutely|definitely)', 'Affirmation'),
            (r'^(here\'s|here are|here is)', 'Here\'s/Here are'),
            (r'^(let me|i\'ll|i will|i can)', 'First person commitment'),
            (r'^(first|firstly|to begin)', 'Sequential starter'),
            (r'^(the\s+\w+\s+(?:is|are))', 'Direct answer'),
            (r'^(to\s+\w+)', 'Infinitive starter'),
            (r'^(yes|no|maybe|perhaps)', 'Yes/No answer'),
            (r'^(i\'m sorry|i apologize)', 'Apology'),
            (r'^(thank you|thanks)', 'Gratitude'),
            (r'^(in\s+(?:summary|conclusion|brief))', 'Summary starter'),
        ]

        for pattern, starter_type in starter_patterns:
            if re.search(pattern, response_start):
                starters.append(starter_type)

        return starters

    def classify_task_type(self, instruction: str, verbs: List[str]) -> str:
        """
        将指令分类到任务类型

        输入:
            instruction: 指令文本
            verbs: 从指令中提取的动词列表
        输出:
            任务类型字符串
        """
        instruction_lower = instruction.lower()

        # 基于关键词的任务分类
        if any(v in ['explain', 'describe', 'define', 'clarify'] for v in verbs):
            return 'explanation'
        elif any(v in ['summarize', 'sum', 'brief', 'outline'] for v in verbs):
            return 'summarization'
        elif any(v in ['translate', 'convert', 'transform'] for v in verbs):
            return 'translation'
        elif any(v in ['create', 'generate', 'write', 'compose', 'design'] for v in verbs):
            return 'generation'
        elif any(v in ['analyze', 'evaluate', 'assess', 'critique'] for v in verbs):
            return 'analysis'
        elif any(v in ['classify', 'categorize', 'identify', 'detect'] for v in verbs):
            return 'classification'
        elif any(v in ['solve', 'calculate', 'compute', 'determine'] for v in verbs):
            return 'problem_solving'
        elif any(v in ['correct', 'fix', 'revise', 'edit'] for v in verbs):
            return 'correction'
        elif 'code' in instruction_lower or 'program' in instruction_lower:
            return 'coding'
        else:
            return 'other'

    def analyze_dataset(self, data: List[Dict]):
        """
        分析整个数据集，提取所有特征

        输入:
            data: 数据集列表
        处理:
            遍历所有样本，提取并统计各类特征
        """
        print("开始分析数据集...")

        for sample in tqdm(data, desc="分析样本"):
            # 获取instruction和response
            instruction = sample.get('prompt', '')
            response = sample.get('generation', '')

            if not instruction or not response:
                continue

            # 1. 提取指令动词
            verbs = self.extract_verbs_from_instruction(instruction)
            for verb in verbs:
                self.instruction_verbs[verb] += 1

            # 2. 提取疑问句模式
            patterns = self.extract_question_patterns(instruction)
            if patterns['question_word']:
                self.question_words[patterns['question_word']] += 1

            # 3. 提取指令模板
            templates = self.extract_instruction_templates(instruction)
            for template in templates:
                self.instruction_patterns[template] += 1

            # 4. 提取回答开头模式
            starters = self.extract_response_starters(response)
            for starter in starters:
                self.response_starters[starter] += 1

            # 5. 任务类型分类
            task_type = self.classify_task_type(instruction, verbs)
            self.task_types[task_type].append({
                'instruction': instruction[:100],  # 只保存前100字符
                'verbs': verbs
            })

    def build_dictionaries(self) -> Dict:
        """
        构建最终的词典

        输出:
            包含三种词典的字典
        """
        print("\n构建最终词典...")

        # 1. 构建指令动词词典
        # 合并预定义动词
        for verb in self.predefined_verbs:
            if verb not in self.instruction_verbs:
                self.instruction_verbs[verb] = 1  # 给预定义动词一个基础频率

        # 获取top-k动词
        top_verbs = dict(self.instruction_verbs.most_common(self.top_k_verbs))

        # 2. 构建指令结构模式库
        top_patterns = dict(self.instruction_patterns.most_common(self.top_k_patterns))

        # 3. 构建回答模式词汇表
        response_patterns = dict(self.response_starters.most_common())

        # 4. 疑问词统计
        question_word_stats = dict(self.question_words.most_common())

        # 5. 任务类型统计
        task_type_stats = {
            task: len(samples) for task, samples in self.task_types.items()
        }

        dictionaries = {
            'instruction_verbs': {
                'verbs': list(top_verbs.keys()),
                'frequencies': top_verbs,
                'total_unique': len(self.instruction_verbs),
                'predefined_included': list(self.predefined_verbs)
            },
            'instruction_patterns': {
                'patterns': list(top_patterns.keys()),
                'frequencies': top_patterns,
                'total_unique': len(self.instruction_patterns)
            },
            'response_starters': {
                'starters': list(response_patterns.keys()),
                'frequencies': response_patterns,
                'total_unique': len(self.response_starters)
            },
            'question_words': {
                'words': list(question_word_stats.keys()),
                'frequencies': question_word_stats
            },
            'task_types': {
                'distribution': task_type_stats,
                'total_samples': sum(task_type_stats.values())
            },
            'statistics': {
                'total_samples_analyzed': sum(task_type_stats.values()),
                'unique_verbs_found': len(self.instruction_verbs),
                'unique_patterns_found': len(self.instruction_patterns),
                'unique_response_starters': len(self.response_starters)
            }
        }

        return dictionaries

    def save_results(self, dictionaries: Dict, output_dir: str):
        """
        保存分析结果

        输入:
            dictionaries: 构建的词典
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. 保存完整词典（JSON格式）
        output_path = os.path.join(output_dir, 'instruction_dictionaries.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dictionaries, f, ensure_ascii=False, indent=2)
        print(f"词典已保存到: {output_path}")

        # 2. 保存动词列表（便于直接使用）
        # verb_list_path = os.path.join(output_dir, 'instruction_verbs.txt')
        # with open(verb_list_path, 'w', encoding='utf-8') as f:
        #     for verb in dictionaries['instruction_verbs']['verbs']:
        #         f.write(f"{verb}\n")
        # print(f"动词列表已保存到: {verb_list_path}")

        # 3. 保存模式列表
        # pattern_list_path = os.path.join(output_dir, 'instruction_patterns.txt')
        # with open(pattern_list_path, 'w', encoding='utf-8') as f:
        #     for pattern in dictionaries['instruction_patterns']['patterns']:
        #         f.write(f"{pattern}\n")
        # print(f"模式列表已保存到: {pattern_list_path}")

        # 4. 保存统计报告
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("指令数据集分析报告\n")
            f.write("=" * 50 + "\n\n")

            f.write("【总体统计】\n")
            f.write(f"分析样本总数: {dictionaries['statistics']['total_samples_analyzed']}\n")
            f.write(f"发现独特动词数: {dictionaries['statistics']['unique_verbs_found']}\n")
            f.write(f"发现独特模式数: {dictionaries['statistics']['unique_patterns_found']}\n")
            f.write(f"发现独特回答开头: {dictionaries['statistics']['unique_response_starters']}\n\n")

            f.write("【Top-10 高频指令动词】\n")
            for i, (verb, freq) in enumerate(list(dictionaries['instruction_verbs']['frequencies'].items())[:10], 1):
                f.write(f"{i}. {verb}: {freq}次\n")
            f.write("\n")

            f.write("【Top-10 指令模式】\n")
            for i, (pattern, freq) in enumerate(list(dictionaries['instruction_patterns']['frequencies'].items())[:10],
                                                1):
                f.write(f"{i}. {pattern}: {freq}次\n")
            f.write("\n")

            f.write("【回答开头模式分布】\n")
            for starter, freq in dictionaries['response_starters']['frequencies'].items():
                f.write(f"- {starter}: {freq}次\n")
            f.write("\n")

            f.write("【任务类型分布】\n")
            for task_type, count in dictionaries['task_types']['distribution'].items():
                percentage = count / dictionaries['task_types']['total_samples'] * 100
                f.write(f"- {task_type}: {count}个 ({percentage:.2f}%)\n")

        print(f"分析报告已保存到: {report_path}")

        # 5. 保存为pickle格式（便于Python直接加载）
        # pickle_path = os.path.join(output_dir, 'instruction_dictionaries.pkl')
        # with open(pickle_path, 'wb') as f:
        #     pickle.dump(dictionaries, f)
        # print(f"Pickle格式已保存到: {pickle_path}")


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='构建指令词典')

    parser.add_argument('--data_path',
                        type=str,
                        default='/root/autodl-tmp/ift_memorization/data/instruction_test_data/combined_instruction_tests.jsonl',
                        help='指令数据集路径')

    parser.add_argument('--output_dir',
                        type=str,
                        default='/root/autodl-tmp/ift_memorization/results/exp1_2/',
                        help='输出目录')

    parser.add_argument('--top_k_verbs',
                        type=int,
                        default=500,
                        help='提取高频动词的数量')

    parser.add_argument('--top_k_patterns',
                        type=int,
                        default=30,
                        help='提取高频句式模板的数量')

    parser.add_argument('--response_prefix_len',
                        type=int,
                        default=100,
                        help='分析response开头的token数')

    args = parser.parse_args()

    # 预定义的指令动词
    predefined_verbs = [
        # 第一组
        "translate", "explain", "summarize", "retrieve",
        "revise", "generate", "describe", "classify", "create",
        "evaluate", "correct", "develop",
        "identify", "analyze", "compose", "demonstrate", "interpret",
        "design", "solve", "follow", "clarify", "say", "help", "act",
        "recommend", "estimate", "edit", "format", "repeat",
        # 第二组
        "write", "give", "find", "create", "make", "describe", "design",
        "generate", "classify", "have", "explain", "tell", "identify",
        "output", "predict", "detect"
    ]

    # 去重
    predefined_verbs = list(set(predefined_verbs))

    # 初始化构建器
    builder = InstructionDictionaryBuilder(
        predefined_verbs=predefined_verbs,
        top_k_verbs=args.top_k_verbs,
        top_k_patterns=args.top_k_patterns,
        response_prefix_len=args.response_prefix_len
    )

    # 加载数据集
    data = builder.load_dataset(args.data_path)

    # 分析数据集
    builder.analyze_dataset(data)

    # 构建词典
    dictionaries = builder.build_dictionaries()

    # 保存结果
    builder.save_results(dictionaries, args.output_dir)

    print("\n词典构建完成！")
    print(f"结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()