#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended similarity analyzer
Includes the original instruction-tuning dataset similarity analysis, plus the newly added:
- Experiment 2: coherence between generated content and prefix content
- Experiment 3: certainty/uncertainty vocabulary analysis
- Experiment 4: full-answer tendency analysis
"""

import json
import argparse
import os
from collections import Counter, defaultdict
import re
from typing import Dict, List, Tuple, Set
import numpy as np
from pathlib import Path
import logging
import math
from scipy.stats import entropy
from WORDS import WORDSSet


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedSimilarityAnalyzer:
    def __init__(self):
        """
        Initialize the enhanced similarity analyzer

        Args:
            instruction_verbs: Instruction verb list
        """
        self.instruction_verbs = WORDSSet.instruction_verbs

        # Politeness marker list
        self.politeness_markers = WORDSSet.politeness_markers

        # Structural marker list
        self.structure_markers = WORDSSet.structure_markers

        # Question word list
        self.question_words = WORDSSet.question_words

        # Experiment 3: certainty/uncertainty vocabulary
        self.certainty_markers = WORDSSet.certainty_markers

        self.uncertainty_markers = WORDSSet.uncertainty_markers

        # Discourse transition markers
        self.transition_markers = WORDSSet.transition_markers

        # Experiment 4: full-answer-related markers
        self.conclusion_markers = WORDSSet.conclusion_markers

        # Task completion markers
        self.task_completion_markers = WORDSSet.task_completion_markers

    def load_instruction_data(self, filepath: str) -> List[Dict]:
        """
        Load the instruction-tuning dataset

        Args:
            filepath: Path to the instruction-tuning dataset

        Returns:
            A list of instruction samples, each containing prompt and generation
        """
        logger.info(f"加载指令微调数据集: {filepath}")
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        logger.info(f"加载了 {len(data)} 条指令数据")
        return data

    def load_generation_data(self, filepath: str) -> List[Dict]:
        """
        Load model generation data

        Args:
            filepath: Path to the model generation data

        Returns:
            A list of generation records
        """
        logger.info(f"加载模型生成数据: {filepath}")
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        logger.info(f"加载了 {len(data)} 条生成数据")
        return data

    def extract_vocabulary(self, texts: List[str]) -> Dict[str, int]:
        """
        Extract word-frequency statistics from texts

        Args:
            texts: List of texts

        Returns:
            Vocabulary frequency dict
        """
        vocab_counter = Counter()
        for text in texts:
            # Simple tokenization into words, lowercased
            words = re.findall(r'\b\w+\b', text.lower())
            vocab_counter.update(words)
        return dict(vocab_counter)

    def calculate_vocabulary_overlap(self, vocab1: Dict[str, int], vocab2: Dict[str, int]) -> Dict[str, float]:
        """
        Compute overlap between two vocabularies

        Args:
            vocab1: First vocabulary
            vocab2: Second vocabulary

        Returns:
            Dict containing various overlap metrics
        """
        set1 = set(vocab1.keys())
        set2 = set(vocab2.keys())

        intersection = set1 & set2
        union = set1 | set2

        # Compute Jaccard similarity
        jaccard = len(intersection) / len(union) if union else 0

        # Compute overlap ratios
        overlap_ratio_1 = len(intersection) / len(set1) if set1 else 0
        overlap_ratio_2 = len(intersection) / len(set2) if set2 else 0

        # Compute frequency correlation on shared words
        common_words = list(intersection)
        if common_words:
            freq1 = [vocab1[word] for word in common_words]
            freq2 = [vocab2[word] for word in common_words]
            correlation = np.corrcoef(freq1, freq2)[0, 1] if len(freq1) > 1 else 0
        else:
            correlation = 0

        return {
            'jaccard_similarity': jaccard,
            'overlap_ratio_1': overlap_ratio_1,
            'overlap_ratio_2': overlap_ratio_2,
            'common_words_count': len(intersection),
            'frequency_correlation': correlation
        }

    def calculate_instruction_density(self, texts: List[str]) -> Dict[str, float]:
        """
        Compute instruction-word density

        Args:
            texts: List of texts

        Returns:
            Instruction density statistics
        """
        total_words = 0
        instruction_word_count = 0
        instruction_text_count = 0

        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            total_words += len(words)

            # Count occurrences of instruction words
            text_has_instruction = False
            for word in words:
                if word in self.instruction_verbs:
                    instruction_word_count += 1
                    text_has_instruction = True

            if text_has_instruction:
                instruction_text_count += 1

        return {
            'instruction_word_density': instruction_word_count / total_words if total_words > 0 else 0,
            'instruction_text_ratio': instruction_text_count / len(texts) if texts else 0,
            'total_instruction_words': instruction_word_count,
            'total_words': total_words
        }

    def analyze_politeness_markers(self, texts: List[str]) -> Dict[str, float]:
        """
        Analyze usage of politeness markers

        Args:
            texts: List of texts

        Returns:
            Politeness marker statistics
        """
        total_texts = len(texts)
        politeness_count = 0
        marker_freq = Counter()

        for text in texts:
            text_lower = text.lower()
            text_has_politeness = False

            for marker in self.politeness_markers:
                if marker in text_lower:
                    marker_freq[marker] += 1
                    text_has_politeness = True

            if text_has_politeness:
                politeness_count += 1

        return {
            'politeness_ratio': politeness_count / total_texts if total_texts > 0 else 0,
            'marker_frequencies': dict(marker_freq),
            'total_marker_count': sum(marker_freq.values())
        }

    def analyze_structure_markers(self, texts: List[str]) -> Dict[str, float]:
        """
        Analyze usage of structural markers

        Args:
            texts: List of texts

        Returns:
            Structural marker statistics
        """
        total_texts = len(texts)
        structured_count = 0
        marker_freq = Counter()

        for text in texts:
            text_lower = text.lower()
            text_has_structure = False

            for marker in self.structure_markers:
                if marker in text_lower:
                    marker_freq[marker] += 1
                    text_has_structure = True

            if text_has_structure:
                structured_count += 1

        return {
            'structured_ratio': structured_count / total_texts if total_texts > 0 else 0,
            'marker_frequencies': dict(marker_freq),
            'total_marker_count': sum(marker_freq.values())
        }

    def analyze_question_patterns(self, texts: List[str]) -> Dict[str, float]:
        """
        Analyze question-sentence patterns

        Args:
            texts: List of texts

        Returns:
            Question pattern statistics
        """
        total_texts = len(texts)
        question_count = 0
        question_word_freq = Counter()

        for text in texts:
            # Check whether a question mark appears
            has_question_mark = '?' in text

            # Check whether it starts with a question word
            words = re.findall(r'\b\w+\b', text.lower())
            has_question_word = False

            if words:
                first_word = words[0]
                if first_word in self.question_words:
                    question_word_freq[first_word] += 1
                    has_question_word = True

            if has_question_mark or has_question_word:
                question_count += 1

        return {
            'question_ratio': question_count / total_texts if total_texts > 0 else 0,
            'question_word_frequencies': dict(question_word_freq),
            'total_question_words': sum(question_word_freq.values())
        }

    # ============= Experiment 2: coherence between generated content and prefix content =============
    def analyze_coherence_with_prefix(self, data: List[Dict]) -> Dict[str, float]:
        """
        Analyze coherence between generated content and the prefix

        Args:
            data: List of records containing prefix and generated_text

        Returns:
            Coherence analysis results
        """
        total_samples = len(data)

        # Lexical coherence stats
        vocab_coherence_scores = []
        # Syntactic coherence stats
        syntax_coherence_scores = []
        # Topic-shift stats
        topic_shift_count = 0
        # Sentence boundary entropy values
        sentence_boundary_entropies = []
        # Transition marker densities
        transition_densities = []

        for item in data:
            prefix = item.get('prefix', '')
            generated = item.get('generated_text', '')

            if not prefix or not generated:
                continue

            # 1. Lexical coherence: proportion of overlapping vocabulary
            prefix_words = set(re.findall(r'\b\w+\b', prefix.lower()))
            generated_words = set(re.findall(r'\b\w+\b', generated.lower()))

            if prefix_words:
                vocab_overlap = len(prefix_words & generated_words) / len(prefix_words)
                vocab_coherence_scores.append(vocab_overlap)

            # 2. Syntactic coherence: analyze sentence-structure similarity
            # Use a more reasonable method: compare POS/functional-word distribution between
            # the last prefix sentence and the first generated sentence
            prefix_sentences = [s.strip() for s in re.split(r'[.!?]', prefix) if s.strip()]
            generated_sentences = [s.strip() for s in re.split(r'[.!?]', generated) if s.strip()]

            if prefix_sentences and generated_sentences:
                last_prefix_sent = prefix_sentences[-1]
                first_generated_sent = generated_sentences[0]

                # Simplified syntactic proxy: compare function-word ratio
                def get_function_word_ratio(sentence):
                    words = re.findall(r'\b\w+\b', sentence.lower())
                    function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                                      'by', 'from', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
                                      'below', 'between', 'among', 'under', 'over', 'he', 'she', 'it', 'they', 'this',
                                      'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                                      'may', 'might', 'can'}
                    if not words:
                        return 0
                    function_count = sum(1 for word in words if word in function_words)
                    return function_count / len(words)

                prefix_ratio = get_function_word_ratio(last_prefix_sent)
                generated_ratio = get_function_word_ratio(first_generated_sent)

                # Similarity = 1 - absolute difference
                syntax_similarity = 1 - abs(prefix_ratio - generated_ratio)
                syntax_coherence_scores.append(syntax_similarity)

            # 3. Topic-shift detection: check for explicit discourse transition markers
            full_text = prefix + " " + generated
            if any(marker in full_text.lower() for marker in self.transition_markers):
                topic_shift_count += 1

            # 4. Sentence boundary entropy: analyze sentence-length variation in generated text
            if generated_sentences and len(generated_sentences) > 1:
                sent_lengths = [len(s.split()) for s in generated_sentences]
                if sent_lengths and max(sent_lengths) > 0:
                    # Compute entropy of the sentence-length distribution
                    length_counts = Counter(sent_lengths)
                    total_sents = len(sent_lengths)
                    length_probs = [count / total_sents for count in length_counts.values()]
                    sent_entropy = -sum(p * math.log2(p) for p in length_probs if p > 0)
                    sentence_boundary_entropies.append(sent_entropy)

            # 5. Transition density: compute density of transition markers in generated text
            generated_words_list = re.findall(r'\b\w+\b', generated.lower())
            if generated_words_list:
                transition_count = sum(1 for marker in self.transition_markers if marker in generated.lower())
                transition_density = transition_count / len(generated_words_list)
                transition_densities.append(transition_density)

        return {
            'avg_vocab_coherence': np.mean(vocab_coherence_scores) if vocab_coherence_scores else 0,
            'avg_syntax_coherence': np.mean(syntax_coherence_scores) if syntax_coherence_scores else 0,
            'topic_shift_ratio': topic_shift_count / total_samples if total_samples > 0 else 0,
            'avg_sentence_boundary_entropy': np.mean(sentence_boundary_entropies) if sentence_boundary_entropies else 0,
            'avg_transition_density': np.mean(transition_densities) if transition_densities else 0,
            'total_analyzed_samples': total_samples
        }

    # ============= Experiment 3: certainty/uncertainty vocabulary analysis =============
    def analyze_certainty_patterns(self, data: List[Dict]) -> Dict[str, float]:
        """
        Analyze certainty/uncertainty vocabulary usage patterns

        Args:
            data: List of records containing generated text (and optionally probability info)

        Returns:
            Certainty-pattern analysis results
        """
        total_texts = len(data)
        certainty_scores = []
        uncertainty_scores = []
        sentence_boundary_entropies = []
        transition_densities = []

        for item in data:
            text = item.get('generated_text', '')

            if not text:
                continue

            words = re.findall(r'\b\w+\b', text.lower())
            total_words = len(words)

            if total_words == 0:
                continue

            # 1. Certainty marker density
            certainty_count = sum(1 for word in words if word in self.certainty_markers)
            certainty_scores.append(certainty_count / total_words)

            # 2. Uncertainty marker density
            uncertainty_count = sum(1 for word in words if word in self.uncertainty_markers)
            uncertainty_scores.append(uncertainty_count / total_words)

            # 3. Sentence boundary entropy (simplified)
            sentences = re.split(r'[.!?]', text)
            if len(sentences) > 1:
                # Compute entropy of the sentence-length distribution
                sent_lengths = [len(s.split()) for s in sentences if s.strip()]
                if sent_lengths:
                    # Normalize length distribution
                    total_length = sum(sent_lengths)
                    if total_length > 0:
                        length_probs = [l / total_length for l in sent_lengths]
                        # Compute entropy
                        sent_entropy = -sum(p * math.log2(p) for p in length_probs if p > 0)
                        sentence_boundary_entropies.append(sent_entropy)

            # 4. Discourse transition density
            transition_count = sum(1 for marker in self.transition_markers if marker in text.lower())
            transition_densities.append(transition_count / total_words)

        return {
            'avg_certainty_density': np.mean(certainty_scores) if certainty_scores else 0,
            'avg_uncertainty_density': np.mean(uncertainty_scores) if uncertainty_scores else 0,
            'avg_sentence_boundary_entropy': np.mean(sentence_boundary_entropies) if sentence_boundary_entropies else 0,
            'avg_transition_density': np.mean(transition_densities) if transition_densities else 0,
            'certainty_uncertainty_ratio': (
                        np.mean(certainty_scores) / np.mean(uncertainty_scores)) if uncertainty_scores and np.mean(
                uncertainty_scores) > 0 else 0
        }

    # ============= Experiment 4: full-answer tendency =============
    def analyze_answer_completeness(self, data: List[Dict]) -> Dict[str, float]:
        """
        Analyze the tendency of generated content to provide complete answers

        Args:
            data: List of records containing generated text

        Returns:
            Full-answer tendency analysis results
        """
        total_texts = len(data)
        completion_scores = []
        task_orientation_scores = []
        pos_diversity_scores = []

        for item in data:
            text = item.get('generated_text', '')

            if not text:
                continue

            words = re.findall(r'\b\w+\b', text.lower())
            total_words = len(words)

            if total_words == 0:
                continue

            # 1. Answer completeness score: check reasonable usage of conclusion markers
            conclusion_count = sum(1 for marker in self.conclusion_markers if marker in text.lower())
            # Normalize completeness score to a reasonable ratio w.r.t. text length
            if total_words > 0:
                # A reasonable range: 1-2 conclusion markers per 100 words
                expected_ratio = total_words / 100 * 0.015  # Baseline ratio: 1.5%
                completion_score = min(conclusion_count / max(expected_ratio, 0.01), 1.0)
            else:
                completion_score = 0
            completion_scores.append(completion_score)

            # 2. Task-orientation: check task-related markers
            task_word_count = sum(1 for word in words if word in self.task_completion_markers)
            task_orientation_scores.append(task_word_count / total_words)

            # 3. POS diversity (a proxy for information density): analyze grammatical category diversity
            # Use simplified POS identification rules
            pos_categories = {
                'nouns': 0,  # Nouns
                'verbs': 0,  # Verbs
                'adjectives': 0,  # Adjectives
                'adverbs': 0,  # Adverbs
                'pronouns': 0,  # Pronouns
                'prepositions': 0  # Prepositions
            }

            # Simplified POS identification rules
            for word in words:
                if word.endswith(('ing', 'ed', 'er', 'est')):
                    if word.endswith('ing'):
                        pos_categories['verbs'] += 1
                    elif word.endswith('ed'):
                        pos_categories['verbs'] += 1
                    elif word.endswith(('er', 'est')):
                        pos_categories['adjectives'] += 1
                elif word.endswith(('ly')):
                    pos_categories['adverbs'] += 1
                elif word in {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'this', 'that', 'these', 'those'}:
                    pos_categories['pronouns'] += 1
                elif word in {'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'about', 'under', 'over',
                              'through', 'during', 'before', 'after'}:
                    pos_categories['prepositions'] += 1
                else:
                    # Default to noun
                    pos_categories['nouns'] += 1

            # Compute POS diversity (Shannon entropy)
            total_pos_count = sum(pos_categories.values())
            if total_pos_count > 0:
                pos_probs = [count / total_pos_count for count in pos_categories.values() if count > 0]
                pos_diversity = -sum(p * math.log2(p) for p in pos_probs if p > 0)
                pos_diversity_scores.append(pos_diversity)
            else:
                pos_diversity_scores.append(0)

        return {
            'avg_completion_score': np.mean(completion_scores) if completion_scores else 0,
            'avg_task_orientation': np.mean(task_orientation_scores) if task_orientation_scores else 0,
            'avg_pos_diversity': np.mean(pos_diversity_scores) if pos_diversity_scores else 0,
            'completeness_ratio': sum(1 for score in completion_scores if score > 0.1) / len(
                completion_scores) if completion_scores else 0
        }

    def compare_similarity(self, base_data: List[Dict], sft_data: List[Dict],
                           instruction_data: List[Dict]) -> Dict:
        """
        Compare similarity between base/SFT generations and instruction data (including all experiments)

        Args:
            base_data: Generation data from the base model
            sft_data: Generation data from the SFT model
            instruction_data: Instruction-tuning data

        Returns:
            Full similarity analysis results
        """
        logger.info("开始进行综合相似性分析...")

        # Extract text content
        base_texts = [item['generated_text'] for item in base_data]
        sft_texts = [item['generated_text'] for item in sft_data]
        instruction_texts = [item['generation'] for item in instruction_data]

        logger.info(f"Base模型生成文本数量: {len(base_texts)}")
        logger.info(f"SFT模型生成文本数量: {len(sft_texts)}")
        logger.info(f"指令数据数量: {len(instruction_texts)}\n")

        results = {}

        # ========== Original analysis ==========
        # 1. Vocabulary-level analysis
        logger.info("进行词汇层面分析, base，sft，instructuion三者见的词汇交集、并集等...")
        base_vocab = self.extract_vocabulary(base_texts)
        sft_vocab = self.extract_vocabulary(sft_texts)
        instruction_vocab = self.extract_vocabulary(instruction_texts)

        results['vocabulary_analysis'] = {
            'base_vs_instruction': self.calculate_vocabulary_overlap(base_vocab, instruction_vocab),
            'sft_vs_instruction': self.calculate_vocabulary_overlap(sft_vocab, instruction_vocab),
            'base_vs_sft': self.calculate_vocabulary_overlap(base_vocab, sft_vocab),
            'vocabulary_sizes': {
                'base': len(base_vocab),
                'sft': len(sft_vocab),
                'instruction': len(instruction_vocab)
            }
        }

        # 2. Instruction-related feature analysis (from instruction perspective)
        logger.info("进行指令相关特征分析...")

        # 2.1 Politeness marker analysis
        logger.info("进行礼貌标记分析...")
        results['politeness_analysis'] = {
            'base': self.analyze_politeness_markers(base_texts),
            'sft': self.analyze_politeness_markers(sft_texts),
            'instruction': self.analyze_politeness_markers(instruction_texts)
        }

        # 2.2 Structural marker analysis
        logger.info("进行结构化标记分析...")
        results['structure_analysis'] = {
            'base': self.analyze_structure_markers(base_texts),
            'sft': self.analyze_structure_markers(sft_texts),
            'instruction': self.analyze_structure_markers(instruction_texts)
        }

        # 2.3 Question pattern analysis
        logger.info("进行疑问句模式分析...")
        results['question_analysis'] = {
            'base': self.analyze_question_patterns(base_texts),
            'sft': self.analyze_question_patterns(sft_texts),
            'instruction': self.analyze_question_patterns(instruction_texts)
        }

        # 3: Certainty/uncertainty vocabulary analysis and transition markers
        logger.info("确定性模式分析...")
        results['certainty_analysis'] = {
            'base': self.analyze_certainty_patterns(base_data),
            'sft': self.analyze_certainty_patterns(sft_data),
            'instruction': self.analyze_certainty_patterns([{'generated_text': text} for text in instruction_texts])
        }

        # 2.5 Full-answer tendency analysis
        logger.info("完整答案倾向性分析...")
        results['completeness_analysis'] = {
            'base': self.analyze_answer_completeness(base_data),
            'sft': self.analyze_answer_completeness(sft_data),
            'instruction': self.analyze_answer_completeness([{'generated_text': text} for text in instruction_texts])
        }

        # ========== New experiment: coherence analysis ==========
        # Experiment 2: coherence between generated content and prefix content
        logger.info("进行实验2：连贯性分析...")
        results['coherence_analysis'] = {
            'base': self.analyze_coherence_with_prefix(base_data),
            'sft': self.analyze_coherence_with_prefix(sft_data)
        }

        logger.info("综合相似性分析完成")
        return results


def main():
    parser = argparse.ArgumentParser(description='指令微调数据集相近性分析')
    parser.add_argument('--instruction_data_path', type=str,
                        default='/root/autodl-tmp/ift_memorization/data/instruction_test_data/combined_instruction_tests.jsonl',
                        help='指令微调数据集路径')
    # parser.add_argument('--base_generation_path', type=str,
    #                     default='/root/autodl-tmp/ift_memorization/results/exp1/stackexchange_prefix16_1B_base_50_samples.jsonl',
    #                     help='base模型生成数据路径')
    # parser.add_argument('--sft_generation_path', type=str,
    #                     default='/root/autodl-tmp/ift_memorization/results/exp1/stackexchange_prefix16_1B_sft_50_samples.jsonl',
    #                     help='SFT模型生成数据路径')
    parser.add_argument('--output_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/results/exp1_politeness',
                        help='输出目录')
    parser.add_argument('--model_scale', type=str, default='32B',
                        help='模型规模标识（用于文件命名）')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='每种条件下的最大样本数，None表示全部')
    parser.add_argument('--generation_length', type=int, default=8, help='8, 16, 128')
    parser.add_argument('--data_type', type=str, default='wiki-fact', help='stackexchange, dclm-privacy, wiki-fact')

    args = parser.parse_args()

    args.base_generation_path = f'/root/autodl-tmp/ift_memorization/results/exp1_generation_{args.generation_length}/{args.data_type}_prefix32_new{args.generation_length}_{args.model_scale}_base_{args.max_samples}_samples.jsonl'
    args.sft_generation_path = f'/root/autodl-tmp/ift_memorization/results/exp1_generation_{args.generation_length}/{args.data_type}_prefix32_new{args.generation_length}_{args.model_scale}_sft_{args.max_samples}_samples.jsonl'

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)


    # Initialize analyzer
    analyzer = EnhancedSimilarityAnalyzer()

    # Load data
    instruction_data = analyzer.load_instruction_data(args.instruction_data_path)
    base_data = analyzer.load_generation_data(args.base_generation_path)
    sft_data = analyzer.load_generation_data(args.sft_generation_path)

    # Run similarity analysis
    results = analyzer.compare_similarity(base_data, sft_data, instruction_data)

    # Print key result summary
    print("\n=== 综合相似性分析结果摘要 ===")

    # Original analysis
    print("【生成的内容与指令微调数据集的特点的相似程度】")
    print(
        f"词汇重叠度 (Base vs 指令数据): {results['vocabulary_analysis']['base_vs_instruction']['jaccard_similarity']:.4f}")
    print(
        f"词汇重叠度 (SFT vs 指令数据): {results['vocabulary_analysis']['sft_vs_instruction']['jaccard_similarity']:.4f}")

    print(f"\n礼貌程度 - Base: {results['politeness_analysis']['base']['politeness_ratio']:.4f}")
    print(f"礼貌程度 - SFT: {results['politeness_analysis']['sft']['politeness_ratio']:.4f}")
    print(f"结构化程度 - Base: {results['structure_analysis']['base']['structured_ratio']:.4f}")
    print(f"结构化程度 - SFT: {results['structure_analysis']['sft']['structured_ratio']:.4f}")
    print(f"疑问词 - Base: {results['question_analysis']['base']['question_ratio']:.4f}")
    print(f"疑问词 - SFT: {results['question_analysis']['sft']['question_ratio']:.4f}")
    print(f"不确定性词汇密度 - Base: {results['certainty_analysis']['base']['avg_uncertainty_density']:.4f}")
    print(f"不确定性词汇密度 - SFT: {results['certainty_analysis']['sft']['avg_uncertainty_density']:.4f}")
    print(f"确定性词汇密度 - Base: {results['certainty_analysis']['base']['avg_certainty_density']:.4f}")
    print(f"确定性词汇密度 - SFT: {results['certainty_analysis']['sft']['avg_certainty_density']:.4f}")
    print(f"转折词密度 - Base: {results['certainty_analysis']['base']['avg_transition_density']:.4f}")
    print(f"转折词密度 - SFT: {results['certainty_analysis']['sft']['avg_transition_density']:.4f}")

    print(f"\n完整性得分 - Base: {results['completeness_analysis']['base']['avg_completion_score']:.4f}")
    print(f"完整性得分 - SFT: {results['completeness_analysis']['sft']['avg_completion_score']:.4f}")
    print(f"任务导向性 - Base: {results['completeness_analysis']['base']['avg_task_orientation']:.4f}")
    print(f"任务导向性 - SFT: {results['completeness_analysis']['sft']['avg_task_orientation']:.4f}")
    print(f"词性多样性 - Base: {results['completeness_analysis']['base']['avg_pos_diversity']:.4f}")
    print(f"词性多样性 - SFT: {results['completeness_analysis']['sft']['avg_pos_diversity']:.4f}")

    # New experiment results
    print("\n【实验2：连贯性分析】")
    print(f"词汇连贯性 - Base: {results['coherence_analysis']['base']['avg_vocab_coherence']:.4f}")
    print(f"词汇连贯性 - SFT: {results['coherence_analysis']['sft']['avg_vocab_coherence']:.4f}")
    print(f"句法连贯性 - Base: {results['coherence_analysis']['base']['avg_syntax_coherence']:.4f}")
    print(f"句法连贯性 - SFT: {results['coherence_analysis']['sft']['avg_syntax_coherence']:.4f}")
    print(f"主题转折比例 - Base: {results['coherence_analysis']['base']['topic_shift_ratio']:.4f}")
    print(f"主题转折比例 - SFT: {results['coherence_analysis']['sft']['topic_shift_ratio']:.4f}")
    print(f"句子边界熵 - Base: {results['coherence_analysis']['base']['avg_sentence_boundary_entropy']:.4f}")
    print(f"句子边界熵 - SFT: {results['coherence_analysis']['sft']['avg_sentence_boundary_entropy']:.4f}")

    # Save results
    output_file = os.path.join(args.output_dir,
                               f'exp1_differences_{args.data_type}_{args.model_scale}_length_{args.generation_length}.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"分析结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
