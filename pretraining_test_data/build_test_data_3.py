"""
构建memorization测试数据（内存优化版本）
使用流式处理避免内存溢出
"""
import json
import os
import re
import random
from typing import Dict, Optional, Iterator
import argparse
from tqdm import tqdm


class MemorizationTestBuilder:
    def __init__(self):
        """初始化测试数据构建器"""
        self.instruction_keywords_1 = [
            "translate", "explain", "summarize", "retrieve", "revise",
            'generate', 'describe', 'classify', 'create', "evaluate",
            "correct", "develop", "identify", "analyze", "compose",
            "demonstrate", "interpret", "design", "solve", "follow",
            "clarify", "say", "help", "act", "recommend", "estimate",
            "edit", "format", "repeat", "provide", "show", "teach",
            "guide", "illustrate", "elaborate", "define", "compare"
        ]

        self.instruction_keywords_2 = [
            "write", "give", "find", "create", "make", "describe",
            "design", "generate", "classify", "have", "explain",
            "tell", "identify", "output", "predict", "detect",
            "list", "name", "state", "determine", "calculate"
        ]

        self.all_instruction_keywords = list(set(
            self.instruction_keywords_1 + self.instruction_keywords_2
        ))

        self.question_words = [
            "what", "how", "why", "when", "where", "who", "which",
            "whose", "whom", "whether", "could", "would", "should",
            "can", "will", "do", "does", "is", "are"
        ]

        self.dialogue_markers = [
            "tell me", "show me", "help me", "teach me", "i need",
            "i want", "could you", "can you", "would you", "please help",
            "i'm looking for", "i would like", "let me know"
        ]

    def split_text_into_chunks(
        self,
        text: str,
        chunk_size: int = 512
    ) -> Optional[Dict]:
        """提取文本最前面的固定长度片段（基于word）"""
        words = text.split()
        if len(words) < chunk_size:
            return None

        chunk_words = words[:chunk_size]
        chunk_text = ' '.join(chunk_words)

        return {
            "text": chunk_text,
            "word_count": len(chunk_words),
            "chunk_idx": 0,
            "start_position": 0,
            "original_text_length": len(words)
        }

    def check_instruction_format(self, text: str) -> Dict:
        """检查文本是否包含指令格式特征"""
        text_lower = text.lower()

        found_keywords = []
        for keyword in self.all_instruction_keywords:
            if re.search(r'\b' + keyword + r'\b', text_lower):
                found_keywords.append(keyword)

        found_questions = []
        for q_word in self.question_words:
            pattern = r'^' + q_word + r'\b'
            if re.search(pattern, text_lower, re.MULTILINE):
                found_questions.append(q_word)

        has_question_mark = '?' in text
        question_mark_count = text.count('?')

        found_dialogue_markers = []
        for marker in self.dialogue_markers:
            if marker in text_lower:
                found_dialogue_markers.append(marker)

        answer_patterns = [
            r"^here is", r"^here are", r"^step \d", r"^first,",
            r"^second,", r"^solution:", r"^answer:", r"^you can",
            r"^let me", r"^i'll help", r"^certainly", r"^of course",
            r"^to\s+(solve|answer|address)", r"^the\s+(answer|solution)\s+is",
            r"^\d+\.", r"^[a-z]\)", r"^in\s+conclusion", r"^finally"
        ]
        found_answer_patterns = []
        for pattern in answer_patterns:
            if re.search(pattern, text_lower, re.MULTILINE):
                found_answer_patterns.append(pattern)

        imperative_count = len(re.findall(r'^[A-Z]?[a-z]*\s*[A-Z][a-z]+', text, re.MULTILINE))

        has_numbered_list = bool(re.search(r'^\d+\.', text, re.MULTILINE))
        has_bullet_list = bool(re.search(r'^[-*•]', text, re.MULTILINE))

        score = (
            len(found_keywords) * 3 +
            len(found_questions) * 2 +
            question_mark_count * 1.5 +
            len(found_dialogue_markers) * 2 +
            len(found_answer_patterns) * 2 +
            (2 if has_numbered_list else 0) +
            (2 if has_bullet_list else 0) +
            min(imperative_count, 3)
        )

        instruction_type = None
        if score > 3:
            if len(found_dialogue_markers) > 0:
                instruction_type = "dialogue"
            elif question_mark_count > 0:
                instruction_type = "question-answer"
            elif len(found_keywords) > 2:
                instruction_type = "task-oriented"
            else:
                instruction_type = "general-instruction"

        return {
            "is_instruction": score > 3,
            "score": score,
            "instruction_type": instruction_type,
            "keywords": found_keywords,
            "questions": found_questions,
            "has_question": has_question_mark,
            "dialogue_markers": found_dialogue_markers,
            "answer_patterns": found_answer_patterns,
            "has_list": has_numbered_list or has_bullet_list
        }

    def check_factual_content(self, text: str) -> Dict:
        """检查文本是否包含事实性内容"""
        years = re.findall(r'\b[12]\d{3}\b', text)
        dates = re.findall(
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            text
        )
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        percentages = re.findall(r'\b\d+(?:\.\d+)?%\b', text)
        currencies = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)

        sentences = text.split('.')
        proper_nouns = []
        for sent in sentences:
            words = sent.strip().split()
            if len(words) > 1:
                for word in words[1:]:
                    if re.match(r'^[A-Z][a-z]+', word):
                        proper_nouns.append(word)

        org_indicators = ['Inc', 'Corp', 'LLC', 'Ltd', 'Company', 'Foundation',
                         'Institute', 'University', 'College', 'Association']
        org_count = sum(1 for indicator in org_indicators if indicator in text)

        location_keywords = [
            "located", "based in", "founded in", "headquarters",
            "situated", "established in", "born in", "died in"
        ]
        location_count = sum(1 for kw in location_keywords if kw in text.lower())

        fact_keywords = [
            "founded", "established", "born", "died", "located",
            "population", "area", "capital", "president", "discovered",
            "invented", "published", "released", "launched", "created",
            "built", "opened", "closed", "graduated", "awarded"
        ]
        found_fact_keywords = [kw for kw in fact_keywords if kw in text.lower()]

        quantity_phrases = re.findall(
            r'\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|thousand|hundred)\b',
            text, re.IGNORECASE
        )

        score = (
            len(years) * 3 +
            len(dates) * 3 +
            min(len(numbers), 10) * 1.5 +
            len(percentages) * 2 +
            len(currencies) * 2 +
            min(len(proper_nouns), 15) +
            org_count * 2 +
            location_count * 2 +
            len(found_fact_keywords) * 2 +
            len(quantity_phrases) * 2
        )

        word_count = len(text.split())
        fact_density = (score / max(word_count, 1)) * 100

        return {
            "is_factual": score > 8 or fact_density > 3,
            "score": score,
            "fact_density": fact_density,
            "years": years[:5],
            "dates_count": len(dates),
            "numbers_count": len(numbers),
            "percentages": percentages[:5],
            "currencies": currencies[:5],
            "proper_nouns_count": len(proper_nouns),
            "organizations": org_count,
            "locations": location_count,
            "fact_keywords": found_fact_keywords[:10],
            "quantity_phrases": quantity_phrases[:5]
        }

    def check_dclm_content(self, text: str) -> Dict:
        """检查DCLM数据集内容特征"""
        text_lower = text.lower()

        code_indicators = [
            'def ', 'class ', 'import ', 'function', 'return ',
            'if __name__', 'public static', 'void main'
        ]
        has_code = any(ind in text_lower for ind in code_indicators)

        academic_keywords = [
            'abstract', 'introduction', 'methodology', 'conclusion',
            'references', 'figure', 'table', 'equation', 'theorem'
        ]
        academic_count = sum(1 for kw in academic_keywords if kw in text_lower)

        forum_indicators = [
            'reply', 'posted', 'wrote:', 'quote:', '@', '#',
            'upvote', 'downvote', 'comment'
        ]
        forum_count = sum(1 for ind in forum_indicators if ind in text_lower)

        article_indicators = [
            'reported', 'announced', 'according to', 'sources say',
            'breaking news', 'update:', 'published'
        ]
        article_count = sum(1 for ind in article_indicators if ind in text_lower)

        content_type = "general"
        max_score = 0

        if has_code:
            content_type = "code"
            max_score = 10
        elif academic_count >= 3:
            content_type = "academic"
            max_score = academic_count
        elif forum_count >= 2:
            content_type = "forum"
            max_score = forum_count
        elif article_count >= 2:
            content_type = "article"
            max_score = article_count

        is_selected = False
        if content_type == "general":
            is_selected = random.random() < 0.1
        else:
            is_selected = random.random() < 0.3

        return {
            "is_selected": is_selected,
            "content_type": content_type,
            "score": max_score,
            "has_code": has_code,
            "academic_indicators": academic_count,
            "forum_indicators": forum_count,
            "article_indicators": article_count
        }

    def stream_process_dataset(
        self,
        input_path: str,
        output_path: str,
        dataset_type: str,
        chunk_size: int = 512,
        seed: int = 42,
        batch_size: int = 1000
    ) -> Dict:
        """
        流式处理数据集，避免内存溢出

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            dataset_type: 数据集类型
            chunk_size: 切分片段大小
            seed: 随机种子
            batch_size: 批处理大小（用于批量写入）
        """
        random.seed(seed)

        stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "selected_chunks": 0,
            "skipped_short": 0,
            "total_samples": 0,
            "instruction_chunks": 0,
            "factual_chunks": 0,
            "general_chunks": 0
        }

        # 先统计总行数（用于进度条）
        print(f"正在统计文件行数...")
        with open(input_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        print(f"\n处理 {dataset_type} 数据集（共 {total_lines:,} 行）...")

        # 批量缓存，减少I/O次数
        sample_batch = []

        # 流式处理
        with open(input_path, "r", encoding="utf-8") as f_in, \
             open(output_path, "w", encoding="utf-8") as f_out:

            for doc_idx, line in enumerate(tqdm(f_in, total=total_lines, desc=f"处理文档")):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                stats["total_documents"] += 1
                text = item.get("text", "")

                if not text or len(text) < 100:
                    stats["skipped_short"] += 1
                    continue

                # 切分文本
                chunk = self.split_text_into_chunks(text, chunk_size=chunk_size)

                if chunk is None:
                    stats["skipped_short"] += 1
                    continue

                stats["total_chunks"] += 1
                chunk_text = chunk["text"]
                selected = False
                selection_reason = "general"
                selection_score = 0

                # 根据数据集类型筛选
                if dataset_type == "stackexchange":
                    instruction_check = self.check_instruction_format(chunk_text)
                    if instruction_check["is_instruction"]:
                        selected = True
                        selection_reason = f"instruction_{instruction_check.get('instruction_type', 'general')}"
                        selection_score = instruction_check["score"]
                        stats["instruction_chunks"] += 1

                elif dataset_type == "wiki":
                    factual_check = self.check_factual_content(chunk_text)
                    if factual_check["is_factual"]:
                        selected = True
                        selection_reason = "factual"
                        selection_score = factual_check["score"]
                        stats["factual_chunks"] += 1

                else:  # dclm或其他
                    dclm_check = self.check_dclm_content(chunk_text)
                    if dclm_check["is_selected"]:
                        selected = True
                        selection_reason = f"dclm_{dclm_check['content_type']}"
                        selection_score = dclm_check["score"]
                        stats["general_chunks"] += 1

                if selected:
                    stats["selected_chunks"] += 1
                    sample = {
                        "text": chunk_text,
                        "word_count": chunk["word_count"],
                        "dataset_type": dataset_type,
                        "selection_reason": selection_reason,
                        "selection_score": selection_score,
                        "doc_idx": doc_idx,
                        "original_idx": item.get("idx", -1)
                    }

                    sample_batch.append(sample)
                    stats["total_samples"] += 1

                    # 批量写入，减少I/O
                    if len(sample_batch) >= batch_size:
                        for s in sample_batch:
                            f_out.write(json.dumps(s, ensure_ascii=False) + "\n")
                        sample_batch = []

            # 写入剩余的样本
            if sample_batch:
                for s in sample_batch:
                    f_out.write(json.dumps(s, ensure_ascii=False) + "\n")

        return stats


def main():
    """主函数：解析参数并执行测试数据构建"""
    parser = argparse.ArgumentParser(description="构建memorization测试数据（内存优化版本）")
    parser.add_argument("--input_dir", type=str, default="../../data/pretraining_test_data", help="输入数据目录")
    parser.add_argument("--output_dir", type=str, default="../../data/pretraining_test_data/mem_test", help="输出目录")
    parser.add_argument("--chunk_size", type=int, default=512, help="文本切分的word长度")
    parser.add_argument("--datasets", nargs="+", default=["stackexchange"], help="要处理的数据集列表")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--batch_size", type=int, default=1000, help="批处理大小")
    args = parser.parse_args()

    builder = MemorizationTestBuilder()
    os.makedirs(args.output_dir, exist_ok=True)

    suffix_map = {
        "stackexchange": "instruction",
        "wiki": "fact",
        "dclm": "privacy"
    }

    for dataset_name in args.datasets:
        print(f"\n{'='*50}")
        print(f"开始处理 {dataset_name} 数据集")
        print(f"{'='*50}")

        input_path = os.path.join(args.input_dir, f"{dataset_name}_top5M.jsonl")

        if not os.path.exists(input_path):
            print(f"警告: 找不到输入文件 {input_path}，跳过")
            continue

        suffix = suffix_map.get(dataset_name, "")
        output_filename = f"{dataset_name}_{suffix}.jsonl"
        output_path = os.path.join(args.output_dir, output_filename)

        # 使用流式处理
        stats = builder.stream_process_dataset(
            input_path=input_path,
            output_path=output_path,
            dataset_type=dataset_name,
            chunk_size=args.chunk_size,
            seed=args.seed,
            batch_size=args.batch_size
        )

        # 打印统计信息
        print(f"\n{dataset_name} 数据集处理统计:")
        print(f"  处理文档数: {stats['total_documents']:,}")
        print(f"  生成片段数: {stats['total_chunks']:,}")
        print(f"  跳过（太短）: {stats['skipped_short']:,}")
        print(f"  选中片段数: {stats['selected_chunks']:,}")
        print(f"  指令格式片段: {stats['instruction_chunks']:,}")
        print(f"  事实内容片段: {stats['factual_chunks']:,}")
        print(f"  通用文本片段: {stats['general_chunks']:,}")
        print(f"  最终测试样本数: {stats['total_samples']:,}")

        # 保存统计信息
        stats_filename = f"{dataset_name}_{suffix}_stats.json"
        stats_path = os.path.join(args.output_dir, stats_filename)

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"成功保存 {stats['total_samples']:,} 个测试样本")
        print(f"统计信息已保存到: {stats_path}")

    print(f"\n{'='*50}")
    print(f"所有数据集处理完成！")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()