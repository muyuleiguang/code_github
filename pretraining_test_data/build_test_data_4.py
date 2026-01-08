"""
Build memorization test data (memory-optimized version + deduplication)
Use streaming processing to avoid OOM, and add N-gram deduplication to prevent data leakage
"""
import json
import os
import re
import random
from typing import Dict, Optional, Set, List
import argparse
from tqdm import tqdm


class MemorizationTestBuilder:
    def __init__(self):
        """Initialize the test data builder"""
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

        # Store N-grams from the SFT dataset for deduplication
        self.sft_ngrams: Set[str] = set()
        self.ngram_size = 13  # Use 13-gram by default for deduplication

    def _get_ngrams(self, text: str, n: int) -> Set[str]:
        """Generate N-grams for a text (word-based)"""
        words = text.split()
        if len(words) < n:
            return set()
        return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))

    def load_sft_ngrams(self, sft_path: str, n: int = 13):
        """
        Load the SFT dataset and build an N-gram set for deduplication
        Note: this may consume significant memory but provides fast lookup
        """
        self.ngram_size = n
        print(f"正在加载 SFT 数据用于去重 ({sft_path})...")
        print(f"使用 {n}-gram Overlap 标准")
        
        if not os.path.exists(sft_path):
            print(f"警告: SFT文件 {sft_path} 不存在，将跳过去重步骤！")
            return

        count = 0
        try:
            with open(sft_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="构建SFT N-grams"):
                    try:
                        item = json.loads(line)
                        # Assume SFT data includes 'messages' (list) or 'text' (str)
                        text_content = ""
                        if "text" in item:
                            text_content = item["text"]
                        elif "messages" in item:
                            # Concatenate all dialogue content
                            text_content = " ".join([m.get("content", "") for m in item["messages"]])
                        
                        if text_content:
                            ngrams = self._get_ngrams(text_content, n)
                            self.sft_ngrams.update(ngrams)
                            count += 1
                    except json.JSONDecodeError:
                        continue
            print(f"SFT N-grams 加载完成。处理样本数: {count}, 唯一 {n}-grams 数: {len(self.sft_ngrams):,}")
        except Exception as e:
            print(f"加载 SFT 数据出错: {e}")

    def check_contamination(self, text: str) -> bool:
        """
        Check whether the text is contaminated (i.e., contains an N-gram from the SFT dataset)
        Return True if contaminated (should be discarded), False if clean
        """
        if not self.sft_ngrams:
            return False
            
        target_ngrams = self._get_ngrams(text, self.ngram_size)
        # Check if there is any intersection
        if not target_ngrams.isdisjoint(self.sft_ngrams):
            return True
        return False

    def split_text_into_chunks(
        self,
        text: str,
        chunk_size: int = 512
    ) -> Optional[Dict]:
        """Extract a fixed-length prefix chunk from the text (word-based)"""
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
        """Check whether the text contains instruction-format features"""
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
        """Check whether the text contains factual content"""
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
        """Check content characteristics of the DCLM dataset"""
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
        Stream-process the dataset to avoid OOM

        Args:
            input_path: Input file path
            output_path: Output file path
            dataset_type: Dataset type
            chunk_size: Chunk size for splitting
            seed: Random seed
            batch_size: Batch size (for batched writing)
        """
        random.seed(seed)

        stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "selected_chunks": 0,
            "contaminated_chunks": 0,  # New: count how many were removed by deduplication
            "skipped_short": 0,
            "total_samples": 0,
            "instruction_chunks": 0,
            "factual_chunks": 0,
            "general_chunks": 0
        }

        # Count total lines first (for progress bar)
        print(f"正在统计文件行数...")
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                total_lines = sum(1 for _ in f)
        except FileNotFoundError:
            print(f"错误: 找不到文件 {input_path}")
            return stats

        print(f"\n处理 {dataset_type} 数据集（共 {total_lines:,} 行）...")

        # Batch buffer to reduce I/O calls
        sample_batch = []

        # Stream processing
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

                # Split text
                chunk = self.split_text_into_chunks(text, chunk_size=chunk_size)

                if chunk is None:
                    stats["skipped_short"] += 1
                    continue

                stats["total_chunks"] += 1
                chunk_text = chunk["text"]
                selected = False
                selection_reason = "general"
                selection_score = 0

                # 1) Apply content-based selection first; if it fails, skip deduplication to save work
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

                else:  # dclm or others
                    dclm_check = self.check_dclm_content(chunk_text)
                    if dclm_check["is_selected"]:
                        selected = True
                        selection_reason = f"dclm_{dclm_check['content_type']}"
                        selection_score = dclm_check["score"]
                        stats["general_chunks"] += 1

                # 2) If selected, run deduplication / contamination check
                if selected:
                    if self.check_contamination(chunk_text):
                        selected = False
                        stats["contaminated_chunks"] += 1
                        # The sample matches content criteria but is discarded due to contamination
                        # print(f"检测到污染，丢弃文档 {doc_idx}")
                        
                # 3) Final write
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

                    # Batch write to reduce I/O
                    if len(sample_batch) >= batch_size:
                        for s in sample_batch:
                            f_out.write(json.dumps(s, ensure_ascii=False) + "\n")
                        sample_batch = []

            # Write remaining samples
            if sample_batch:
                for s in sample_batch:
                    f_out.write(json.dumps(s, ensure_ascii=False) + "\n")

        return stats


def main():
    """Main entry: parse args and run test data construction"""
    parser = argparse.ArgumentParser(description="构建memorization测试数据（内存优化版本 + 去重）")
    parser.add_argument("--input_dir", type=str, default="../../data/pretraining_test_data", help="输入数据目录")
    parser.add_argument("--output_dir", type=str, default="../../data/pretraining_test_data/mem_test", help="输出目录")
    parser.add_argument("--sft_data_path", type=str, default="", help="SFT数据集路径 (jsonl格式)，用于去重/污染检查。如果不传则不去重。")
    parser.add_argument("--chunk_size", type=int, default=512, help="文本切分的word长度")
    parser.add_argument("--datasets", nargs="+", default=["stackexchange"], help="要处理的数据集列表")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--batch_size", type=int, default=1000, help="批处理大小")
    parser.add_argument("--ngram_size", type=int, default=13, help="N-gram size for deduplication (default: 13)")
    args = parser.parse_args()

    builder = MemorizationTestBuilder()
    
    # Load SFT data (if provided)
    if args.sft_data_path:
        print(f"\n{'='*50}")
        print("初始化去重模块...")
        builder.load_sft_ngrams(args.sft_data_path, n=args.ngram_size)
        print(f"{'='*50}")
    else:
        print(f"\n警告: 未指定 --sft_data_path，将不进行污染检查！")

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

        # Use streaming processing
        stats = builder.stream_process_dataset(
            input_path=input_path,
            output_path=output_path,
            dataset_type=dataset_name,
            chunk_size=args.chunk_size,
            seed=args.seed,
            batch_size=args.batch_size
        )

        # Print statistics
        print(f"\n{dataset_name} 数据集处理统计:")
        print(f"  处理文档数: {stats['total_documents']:,}")
        print(f"  生成片段数: {stats['total_chunks']:,}")
        print(f"  跳过（太短）: {stats['skipped_short']:,}")
        print(f"  因污染被丢弃: {stats['contaminated_chunks']:,} (N-gram={args.ngram_size})")
        print(f"  选中片段数: {stats['selected_chunks']:,}")
        print(f"  指令格式片段: {stats['instruction_chunks']:,}")
        print(f"  事实内容片段: {stats['factual_chunks']:,}")
        print(f"  通用文本片段: {stats['general_chunks']:,}")
        print(f"  最终测试样本数: {stats['total_samples']:,}")

        # Save statistics
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
