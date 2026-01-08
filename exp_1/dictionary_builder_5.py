#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build instruction dictionaries: extract instruction feature dictionaries from the Tulu3 instruction-tuning dataset
Including: an instruction verb lexicon, an instruction structural pattern library, and a response pattern glossary
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

# Download required NLTK resources
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
    """Instruction dictionary builder"""

    def __init__(self,
                 predefined_verbs: List[str] = None,
                 top_k_verbs: int = 500,
                 top_k_patterns: int = 30,
                 response_prefix_len: int = 100):
        """
        Initialize the builder

        Args:
            predefined_verbs: A predefined list of instruction verbs
            top_k_verbs: Number of high-frequency verbs to extract (default: 500)
            top_k_patterns: Number of high-frequency template patterns to extract (default: 30)
            response_prefix_len: Number of leading tokens to analyze from the response (default: 100)
        """
        self.lemmatizer = WordNetLemmatizer()
        self.top_k_verbs = top_k_verbs
        self.top_k_patterns = top_k_patterns
        self.response_prefix_len = response_prefix_len

        # Predefined instruction verbs
        self.predefined_verbs = set(predefined_verbs) if predefined_verbs else set()

        # Storage for analysis results
        self.instruction_verbs = Counter()  # Instruction verb frequencies
        self.question_words = Counter()  # Question word frequencies
        self.instruction_patterns = Counter()  # Instruction sentence pattern frequencies
        self.response_starters = Counter()  # Response starter pattern frequencies
        self.task_types = defaultdict(list)  # Task-type buckets

    def load_dataset(self, data_path: str) -> List[Dict]:
        """
        Load the instruction dataset

        Input:
            data_path: Path to a JSONL dataset file
        Output:
            A list of samples; each sample contains instruction and response fields
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
        Extract verbs from an instruction

        Input:
            instruction: Instruction text
        Output:
            A list of lemmatized verbs
        Processing:
            1. Tokenization and POS tagging
            2. Extract verbs (VB, VBP, VBZ, etc.)
            3. Lemmatize to base form
        """
        tokens = word_tokenize(instruction.lower())
        pos_tags = pos_tag(tokens)

        verbs = []
        for word, pos in pos_tags:
            # VB: base form, VBP: present tense, VBZ: 3rd person singular present
            # VBG: present participle, VBN: past participle, VBD: past tense
            if pos.startswith('VB'):
                verb_lemma = self.lemmatizer.lemmatize(word, 'v')
                verbs.append(verb_lemma)

        return verbs

    def extract_question_patterns(self, instruction: str) -> Dict[str, any]:
        """
        Extract question patterns and structure

        Input:
            instruction: Instruction text
        Output:
            A dict containing information such as question word and sentence type
        """
        instruction_lower = instruction.lower().strip()

        patterns = {
            'is_question': instruction.endswith('?'),
            'question_word': None,
            'sentence_type': 'statement',  # statement, question, request
            'has_please': 'please' in instruction_lower,
            'has_could_you': 'could you' in instruction_lower or 'can you' in instruction_lower,
        }

        # Extract leading question word
        question_words = ['what', 'how', 'why', 'where', 'when', 'who', 'which', 'whose']
        for qw in question_words:
            if instruction_lower.startswith(qw):
                patterns['question_word'] = qw
                patterns['sentence_type'] = 'question'
                break

        # Detect request-like phrasing
        request_patterns = ['please', 'could you', 'can you', 'would you', 'will you']
        for rp in request_patterns:
            if rp in instruction_lower:
                patterns['sentence_type'] = 'request'
                break

        return patterns

    def extract_instruction_templates(self, instruction: str) -> List[str]:
        """
        Extract instruction templates

        Input:
            instruction: Instruction text
        Output:
            A list of matched templates
        Processing:
            Use regular expressions to match common instruction templates
        """
        templates = []
        instruction_lower = instruction.lower()

        # Define common instruction template patterns
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
        Extract response starter patterns

        Input:
            response: Response text
            max_words: Maximum number of leading words to consider
        Output:
            A list of starter pattern labels
        """
        starters = []
        response_lower = response.lower().strip()

        # Take the first few words
        words = response_lower.split()[:max_words]
        response_start = ' '.join(words)

        # Detect common starter patterns
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
        Classify an instruction into a task type

        Input:
            instruction: Instruction text
            verbs: List of verbs extracted from the instruction
        Output:
            Task type string
        """
        instruction_lower = instruction.lower()

        # Keyword-based task classification
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
        Analyze the full dataset and extract all features

        Input:
            data: List of dataset samples
        Processing:
            Iterate through all samples and extract/statistically aggregate various features
        """
        print("开始分析数据集...")

        for sample in tqdm(data, desc="分析样本"):
            # Get instruction and response
            instruction = sample.get('prompt', '')
            response = sample.get('generation', '')

            if not instruction or not response:
                continue

            # 1. Extract instruction verbs
            verbs = self.extract_verbs_from_instruction(instruction)
            for verb in verbs:
                self.instruction_verbs[verb] += 1

            # 2. Extract question patterns
            patterns = self.extract_question_patterns(instruction)
            if patterns['question_word']:
                self.question_words[patterns['question_word']] += 1

            # 3. Extract instruction templates
            templates = self.extract_instruction_templates(instruction)
            for template in templates:
                self.instruction_patterns[template] += 1

            # 4. Extract response starter patterns
            starters = self.extract_response_starters(response)
            for starter in starters:
                self.response_starters[starter] += 1

            # 5. Task-type classification
            task_type = self.classify_task_type(instruction, verbs)
            self.task_types[task_type].append({
                'instruction': instruction[:100],  # Only keep the first 100 characters
                'verbs': verbs
            })

    def build_dictionaries(self) -> Dict:
        """
        Build the final dictionaries

        Output:
            A dict containing three types of dictionaries
        """
        print("\n构建最终词典...")

        # 1. Build the instruction verb dictionary
        # Merge predefined verbs
        for verb in self.predefined_verbs:
            if verb not in self.instruction_verbs:
                self.instruction_verbs[verb] = 1  # Assign a base frequency to predefined verbs

        # Get top-k verbs
        top_verbs = dict(self.instruction_verbs.most_common(self.top_k_verbs))

        # 2. Build the instruction structure pattern library
        top_patterns = dict(self.instruction_patterns.most_common(self.top_k_patterns))

        # 3. Build the response pattern glossary
        response_patterns = dict(self.response_starters.most_common())

        # 4. Question word statistics
        question_word_stats = dict(self.question_words.most_common())

        # 5. Task-type statistics
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
        Save analysis results

        Input:
            dictionaries: Built dictionaries
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save full dictionaries (JSON format)
        output_path = os.path.join(output_dir, 'instruction_dictionaries.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dictionaries, f, ensure_ascii=False, indent=2)
        print(f"词典已保存到: {output_path}")

        # 2. Save verb list (for direct use)
        # verb_list_path = os.path.join(output_dir, 'instruction_verbs.txt')
        # with open(verb_list_path, 'w', encoding='utf-8') as f:
        #     for verb in dictionaries['instruction_verbs']['verbs']:
        #         f.write(f"{verb}\n")
        # print(f"动词列表已保存到: {verb_list_path}")

        # 3. Save pattern list
        # pattern_list_path = os.path.join(output_dir, 'instruction_patterns.txt')
        # with open(pattern_list_path, 'w', encoding='utf-8') as f:
        #     for pattern in dictionaries['instruction_patterns']['patterns']:
        #         f.write(f"{pattern}\n")
        # print(f"模式列表已保存到: {pattern_list_path}")

        # 4. Save a statistics report
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

        # 5. Save as a pickle file (for direct Python loading)
        # pickle_path = os.path.join(output_dir, 'instruction_dictionaries.pkl')
        # with open(pickle_path, 'wb') as f:
        #     pickle.dump(dictionaries, f)
        # print(f"Pickle格式已保存到: {pickle_path}")


def main():
    # Set command-line arguments
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

    # Predefined instruction verbs
    predefined_verbs = [
        # Group 1
        "translate", "explain", "summarize", "retrieve",
        "revise", "generate", "describe", "classify", "create",
        "evaluate", "correct", "develop",
        "identify", "analyze", "compose", "demonstrate", "interpret",
        "design", "solve", "follow", "clarify", "say", "help", "act",
        "recommend", "estimate", "edit", "format", "repeat",
        # Group 2
        "write", "give", "find", "create", "make", "describe", "design",
        "generate", "classify", "have", "explain", "tell", "identify",
        "output", "predict", "detect"
    ]

    # Remove duplicates
    predefined_verbs = list(set(predefined_verbs))

    # Initialize builder
    builder = InstructionDictionaryBuilder(
        predefined_verbs=predefined_verbs,
        top_k_verbs=args.top_k_verbs,
        top_k_patterns=args.top_k_patterns,
        response_prefix_len=args.response_prefix_len
    )

    # Load dataset
    data = builder.load_dataset(args.data_path)

    # Analyze dataset
    builder.analyze_dataset(data)

    # Build dictionaries
    dictionaries = builder.build_dictionaries()

    # Save results
    builder.save_results(dictionaries, args.output_dir)

    print("\n词典构建完成！")
    print(f"结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
