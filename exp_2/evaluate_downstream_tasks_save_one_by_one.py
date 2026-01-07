#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Few-shot标准化模型评估框架
修复了以下关键问题:
1. 默认配置修改为cuda设备映射和float32精度
2. 增加了详细的数据保存功能(输入输出和指标)
3. 修复了OOM问题和内存管理
4. 修复了GSM8K评估问题(增加max_new_tokens，优化数值提取)
5. 修复了PopQA评估过于宽松的问题
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
import math
from tqdm import tqdm
import warnings
from datetime import datetime
import random
import gc

# 导入提示模板
from prompt_templates import get_few_shot_prompts, format_few_shot_string

warnings.filterwarnings("ignore")


class DatasetLoader:
    """数据集加载器，支持规范化数据格式和可靠的缓存机制"""

    def __init__(self, data_dir: str = "../../data/downstream_test_data", cache_subdir: str = "cached_datasets"):
        self.data_directory = data_dir
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / cache_subdir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_file(self, dataset_name: str, num_samples: int) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{dataset_name}_{num_samples}_samples.json"

    def _save_cached_data(self, data: List[Dict], dataset_name: str, num_samples: int):
        """
        保存规范化的缓存数据
        输入：数据列表，数据集名称，样本数量
        输出：保存到指定路径的JSON文件
        """
        cache_file = self._get_cache_file(dataset_name, num_samples)
        cache_data = {
            'dataset_name': dataset_name,
            'num_samples': len(data),
            'requested_samples': num_samples,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            print(f"✓ 缓存数据已保存: {cache_file}")
        except Exception as e:
            print(f"✗ 缓存保存失败: {e}")

    def _load_cached_data(self, dataset_name: str, num_samples: int) -> Optional[List[Dict]]:
        """
        加载缓存的数据
        输入：数据集名称，样本数量
        输出：缓存的数据列表或None
        """
        cache_file = self._get_cache_file(dataset_name, num_samples)
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)

                # 验证缓存数据的完整性
                if (cached.get('dataset_name') == dataset_name and
                        cached.get('requested_samples') == num_samples and
                        'data' in cached):
                    print(f"✓ 从缓存加载数据: {cache_file}")
                    return cached['data']
                else:
                    print(f"✗ 缓存数据不匹配，重新加载")

            except Exception as e:
                print(f"✗ 缓存文件损坏，重新加载: {e}")
        return None

    def load_mmlu(self, num_samples: int = 200) -> List[Dict]:
        """加载MMLU数据集 - 多任务语言理解"""
        # 检查缓存
        cached_data = self._load_cached_data('mmlu', num_samples)
        if cached_data:
            return cached_data

        print(f"正在加载MMLU数据集，样本数量: {num_samples}")
        try:
            # 加载数据集并确保缓存到指定目录
            test_dataset = load_dataset("cais/mmlu", "all", split="test", cache_dir=self.data_directory,
                                        trust_remote_code=True)

            print(f"MMLU测试集总样本数: {len(test_dataset)}")

            samples = []
            subjects = list(set([item['subject'] for item in test_dataset]))
            print(f"MMLU包含学科数: {len(subjects)}")

            # 按学科平均分配样本
            samples_per_subject = max(1, num_samples // len(subjects))
            remaining_samples = num_samples % len(subjects)

            for i, subject in enumerate(subjects):
                subject_items = [item for item in test_dataset if item['subject'] == subject]

                # 为前几个学科分配额外样本
                current_samples = samples_per_subject + (1 if i < remaining_samples else 0)
                current_samples = min(current_samples, len(subject_items))

                # 随机选择该学科的样本
                if len(subject_items) > current_samples:
                    selected_items = random.sample(subject_items, current_samples)
                else:
                    selected_items = subject_items

                for item in selected_items:
                    if len(samples) >= num_samples:
                        break

                    # 构建标准MMLU格式
                    choices = item['choices']
                    question_text = item['question']
                    formatted_question = f"{question_text}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"

                    samples.append({
                        'id': len(samples),
                        'subject': subject,
                        'question': question_text,
                        'formatted_question': formatted_question,
                        'choices': choices,
                        'answer_letter': ['A', 'B', 'C', 'D'][item['answer']],
                        'answer_index': item['answer'],
                        'dataset': 'mmlu',
                        'evaluation_type': 'multiple_choice_qa'
                    })

                if len(samples) >= num_samples:
                    break

            print(f"✓ MMLU数据集加载完成，共{len(samples)}个样本，覆盖{len(set([s['subject'] for s in samples]))}个学科")

            # 保存到缓存
            self._save_cached_data(samples, 'mmlu', num_samples)
            return samples

        except Exception as e:
            print(f"✗ MMLU数据集加载失败: {e}")
            return []

    def load_gsm8k(self, num_samples: int = 100) -> List[Dict]:
        """加载GSM8K数据集 - 数学推理"""
        cached_data = self._load_cached_data('gsm8k', num_samples)
        if cached_data:
            return cached_data

        print(f"正在加载GSM8K数据集，样本数量: {num_samples}")
        try:
            dataset = load_dataset("openai/gsm8k", "main", split="test", cache_dir=self.data_directory)
            print(f"GSM8K测试集总样本数: {len(dataset)}")

            samples = []

            # 随机选择样本
            if len(dataset) > num_samples:
                indices = random.sample(range(len(dataset)), num_samples)
                selected_items = [dataset[i] for i in indices]
            else:
                selected_items = list(dataset)

            for item in selected_items:
                # 提取数值答案 - GSM8K标准格式
                answer_text = item['answer']
                answer_match = re.search(r'####\s*(.+)', answer_text)
                final_answer = answer_match.group(1).strip() if answer_match else ""

                samples.append({
                    'id': len(samples),
                    'question': item['question'],
                    'answer': final_answer,
                    'solution': answer_text,
                    'dataset': 'gsm8k',
                    'evaluation_type': 'math_problem_solving'
                })

            print(f"✓ GSM8K数据集加载完成，共{len(samples)}个样本")
            self._save_cached_data(samples, 'gsm8k', num_samples)
            return samples

        except Exception as e:
            print(f"✗ GSM8K数据集加载失败: {e}")
            return []

    def load_math(self, num_samples: int = 50) -> List[Dict]:
        """加载MATH数据集 - 高级数学竞赛题"""
        cached_data = self._load_cached_data('math', num_samples)
        if cached_data:
            return cached_data

        print(f"正在加载MATH数据集，样本数量: {num_samples}")
        try:
            dataset = load_dataset("qwedsacf/competition_math", split="train", cache_dir=self.data_directory)
            print(f"MATH测试集总样本数: {len(dataset)}")

            samples = []

            if len(dataset) > num_samples:
                indices = random.sample(range(len(dataset)), num_samples)
                selected_items = [dataset[i] for i in indices]
            else:
                selected_items = list(dataset)

            for item in selected_items:
                samples.append({
                    'id': len(samples),
                    'problem': item['problem'],
                    'solution': item['solution'],
                    'level': item['level'],
                    'type': item['type'],
                    'dataset': 'math',
                    'evaluation_type': 'competition_math'
                })

            print(f"✓ MATH数据集加载完成，共{len(samples)}个样本")
            self._save_cached_data(samples, 'math', num_samples)
            return samples

        except Exception as e:
            print(f"✗ MATH数据集加载失败: {e}")
            return []

    def load_popqa(self, num_samples: int = 100) -> List[Dict]:
        """加载PopQA数据集 - 流行文化问答"""
        cached_data = self._load_cached_data('popqa', num_samples)
        if cached_data:
            return cached_data

        print(f"正在加载PopQA数据集，样本数量: {num_samples}")
        try:
            dataset = load_dataset("akariasai/PopQA", split="test", cache_dir=self.data_directory,
                                   trust_remote_code=True)
            samples = []

            if len(dataset) > num_samples:
                indices = random.sample(range(len(dataset)), num_samples)
                selected_items = [dataset[i] for i in indices]
            else:
                selected_items = list(dataset)

            for item in selected_items:
                samples.append({
                    'id': len(samples),
                    'question': item['question'],
                    'answer': item['possible_answers'][0] if item['possible_answers'] else '',
                    'possible_answers': item['possible_answers'],
                    'dataset': 'popqa',
                    'evaluation_type': 'open_ended_qa'
                })

            print(f"✓ PopQA数据集加载完成，共{len(samples)}个样本")
            self._save_cached_data(samples, 'popqa', num_samples)
            return samples

        except Exception as e:
            print(f"✗ PopQA数据集加载失败: {e}")
            return []

    def load_alpaca_eval(self, num_samples: int = 100) -> List[Dict]:
        """加载AlpacaEval 2.0数据集 - 对话质量评估"""
        cached_data = self._load_cached_data('alpaca_eval2', num_samples)
        if cached_data:
            return cached_data

        print(f"正在加载AlpacaEval2数据集，样本数量: {num_samples}")
        try:
            dataset = load_dataset("lkevinzc/alpaca_eval2", split="eval", cache_dir=self.data_directory)
            samples = []

            if len(dataset) > num_samples:
                indices = random.sample(range(len(dataset)), num_samples)
                selected_items = [dataset[i] for i in indices]
            else:
                selected_items = list(dataset)

            for item in selected_items:
                samples.append({
                    'id': len(samples),
                    'instruction': item['instruction'],
                    'output': item.get('output', ''),
                    'generator': item.get('generator', ''),
                    'dataset': 'alpaca_eval2',
                    'evaluation_type': 'helpfulness'
                })

            print(f"✓ AlpacaEval数据集加载完成，共{len(samples)}个样本")
            self._save_cached_data(samples, 'alpaca_eval2', num_samples)
            return samples

        except Exception as e:
            print(f"✗ AlpacaEval数据集加载失败: {e}")
            return []

    def load_ifeval(self, num_samples: int = 100) -> List[Dict]:
        """加载IFEval数据集 - 指令遵循评估"""
        cached_data = self._load_cached_data('ifeval', num_samples)
        if cached_data:
            return cached_data

        print(f"正在加载IFEval数据集，样本数量: {num_samples}")
        try:
            dataset = load_dataset("google/IFEval", split="train", cache_dir=self.data_directory,
                                   trust_remote_code=True)
            samples = []

            if len(dataset) > num_samples:
                indices = random.sample(range(len(dataset)), num_samples)
                selected_items = [dataset[i] for i in indices]
            else:
                selected_items = list(dataset)

            for item in selected_items:
                samples.append({
                    'id': len(samples),
                    'prompt': item['prompt'],
                    'instruction_id_list': item.get('instruction_id_list', []),
                    'kwargs': item.get('kwargs', {}),
                    'dataset': 'ifeval',
                    'evaluation_type': 'instruction_following'
                })

            print(f"✓ IFEval数据集加载完成，共{len(samples)}个样本")
            self._save_cached_data(samples, 'ifeval', num_samples)
            return samples

        except Exception as e:
            print(f"✗ IFEval数据集加载失败: {e}")
            return []


class BatchModelEvaluator:
    """
    批处理模型评估器，修复了OOM问题和内存管理
    """

    def __init__(self, model_path: str, max_new_tokens: int = 256, batch_size: int = 4,
                 dtype: str = "float32", device_map: str = "cuda"):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens  # 增加默认值以支持数学推理
        self.initial_batch_size = batch_size
        self.current_batch_size = batch_size  # 动态调整的批大小
        self.dtype = dtype
        self.device_map = device_map
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load_model(self) -> bool:
        """
        加载模型和分词器，使用优化的精度和设备配置
        修复了默认配置问题
        """
        print(f"正在加载模型: {self.model_path}")
        print(f"配置 - 精度: {self.dtype}, 批大小: {self.current_batch_size}, 设备映射: {self.device_map}")

        try:
            # 智能处理模型路径
            model_path = self._resolve_model_path(self.model_path)
            print(f"解析后的模型路径: {model_path}")

            # 检查是否为本地路径
            is_local_path = os.path.exists(model_path)
            print(f"是否为本地路径: {is_local_path}")

            # 加载分词器
            print("正在加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left",  # 批处理需要左填充
                local_files_only=is_local_path,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✓ 分词器加载成功")

            # 设置数据类型
            dtype = getattr(torch, self.dtype) if hasattr(torch, self.dtype) else torch.float32
            print(f"使用数据类型: {dtype}")

            # 加载模型
            print("正在加载模型权重...")

            # 针对不同情况使用不同的加载参数
            load_kwargs = {
                "dtype": dtype,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "local_files_only": is_local_path,
            }

            # 只在CUDA可用时设置device_map
            if torch.cuda.is_available() and self.device_map == "cuda":
                load_kwargs["device_map"] = "auto"  # 让transformers自动分配
            elif self.device_map == "cpu":
                load_kwargs["device_map"] = "cpu"

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs
            )

            # 如果没有使用device_map，手动移动模型到设备
            if not torch.cuda.is_available() or self.device_map == "cpu":
                self.model = self.model.to(self.device)

            # 设置为评估模式
            self.model.eval()

            print(f"✓ 模型加载成功: {model_path}")
            print(f"模型精度: {self.model.dtype}")
            print(f"模型设备: {next(self.model.parameters()).device}")
            print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()) / 1e9:.1f}B")

            # 清理内存
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

            return True

        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            print(f"错误类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False

    def _resolve_model_path(self, raw_path: str) -> str:
        """智能解析模型路径，支持多种路径格式"""
        # 如果是绝对路径且存在，直接使用
        if os.path.isabs(raw_path) and os.path.exists(raw_path):
            return raw_path

        # 处理特定的路径格式
        if raw_path.startswith("/root/autodl-tmp/ift_memorization/model_cache/"):
            # 提取模型名称部分
            model_name = os.path.basename(raw_path)

            # 先尝试作为本地路径
            if os.path.exists(raw_path):
                return raw_path

            # 如果本地路径不存在，尝试转换为HuggingFace格式
            if model_name.startswith("allenai_"):
                hf_name = model_name.replace("allenai_", "allenai/")
                print(f"尝试使用HuggingFace格式: {hf_name}")
                return hf_name

        # 处理其他格式
        if raw_path.startswith("allenai_"):
            return raw_path.replace("allenai_", "allenai/")

        # 默认返回原路径
        return raw_path

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        批量生成响应，修复了OOM问题
        增加了动态批大小调整和内存管理
        """
        if self.model is None or self.tokenizer is None:
            return [""] * len(prompts)

        responses = []
        current_batch_size = self.current_batch_size

        # 按批处理大小分组
        for i in tqdm(range(0, len(prompts), current_batch_size), desc="批量生成"):
            batch_prompts = prompts[i:i + current_batch_size]

            # 尝试批量生成，如果OOM则减小批大小
            batch_responses = self._generate_batch_with_fallback(batch_prompts)
            responses.extend(batch_responses)

            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        return responses

    def _generate_batch_with_fallback(self, batch_prompts: List[str]) -> List[str]:
        """
        带OOM回退的批量生成
        """
        current_size = len(batch_prompts)

        while current_size > 0:
            try:
                if current_size == len(batch_prompts):
                    # 尝试批量处理
                    return self._generate_single_batch(batch_prompts)
                else:
                    # 分割成更小的批次
                    responses = []
                    for i in range(0, len(batch_prompts), current_size):
                        sub_batch = batch_prompts[i:i + current_size]
                        sub_responses = self._generate_single_batch(sub_batch)
                        responses.extend(sub_responses)
                    return responses

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    current_size = max(1, current_size // 2)
                    print(f"OOM错误，减小批大小到: {current_size}")
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                    continue
                else:
                    raise e
            except Exception as e:
                print(f"批量生成出错: {e}")
                # 回退到逐个生成
                responses = []
                for prompt in batch_prompts:
                    try:
                        response = self._generate_single_batch([prompt])
                        responses.extend(response)
                    except Exception as e2:
                        print(f"单个生成也失败: {e2}")
                        responses.append("")
                return responses

        return [""] * len(batch_prompts)

    def _generate_single_batch(self, batch_prompts: List[str]) -> List[str]:
        """
        执行单个批次的生成
        """
        # 批量编码
        inputs = self.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 批量生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # 贪婪解码保证可重复性
                temperature=1.0,  # 设置为1.0确保贪婪解码
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        # 解码批量输出
        responses = []
        for j, output in enumerate(outputs):
            # 只保留新生成的部分
            input_length = inputs['input_ids'][j].shape[0]
            generated_tokens = output[input_length:]

            generated_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            responses.append(generated_text.strip())

        return responses


class StandardMetricCalculator:
    """
    标准评估指标计算器，修复了GSM8K和PopQA的评估问题
    """

    @staticmethod
    def evaluate_mmlu(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """MMLU标准评估 - 精确匹配选择题答案"""
        if len(predictions) != len(references):
            return {'accuracy': 0.0, 'total': 0}

        correct = 0
        total = len(predictions)

        for pred, ref in zip(predictions, references):
            pred_clean = StandardMetricCalculator.extract_choice_answer(pred)
            ref_clean = ref.strip().upper()

            if pred_clean == ref_clean:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'metric_name': 'exact_match_accuracy'
        }

    @staticmethod
    def extract_choice_answer(response: str) -> str:
        """
        从模型响应中提取选择题答案，增强了匹配模式
        """
        response = response.strip().upper()

        # 策略1：查找单独的选项字母
        choice_pattern = r'\b([ABCD])\b'
        matches = re.findall(choice_pattern, response)
        if matches:
            return matches[0]

        # 策略2：查找"答案是X"等模式
        answer_patterns = [
            r'ANSWER IS\s*([ABCD])',
            r'ANSWER:\s*([ABCD])',
            r'THE ANSWER IS\s*([ABCD])',
            r'\(([ABCD])\)',
            r'OPTION\s*([ABCD])',
            r'CHOICE\s*([ABCD])',
            r'([ABCD])\)',  # 新增模式
            r'([ABCD])\.',  # 新增模式
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)

        # 策略3：如果响应以选项字母开头
        if response and response[0] in 'ABCD':
            return response[0]

        # 默认返回A（最保守的选择）
        return "A"

    @staticmethod
    def evaluate_gsm8k(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        GSM8K标准评估 - 修复了数值提取问题
        """
        if len(predictions) != len(references):
            return {'accuracy': 0.0, 'total': 0}

        correct = 0
        total = len(predictions)

        for pred, ref in zip(predictions, references):
            pred_num = StandardMetricCalculator.extract_final_number(pred)
            ref_num = StandardMetricCalculator.normalize_number(ref)

            if pred_num and ref_num and StandardMetricCalculator.numbers_equal(pred_num, ref_num):
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'metric_name': 'numeric_accuracy'
        }

    @staticmethod
    def extract_final_number(response: str) -> Optional[str]:
        """
        从响应中提取最终数值答案 - 增强了提取模式
        """
        # 扩展的数值匹配模式
        patterns = [
            r'the answer is\s*([\d,\.\-]+)',
            r'answer:\s*([\d,\.\-]+)',
            r'final answer:\s*([\d,\.\-]+)',
            r'therefore,?\s*([\d,\.\-]+)',
            r'thus,?\s*([\d,\.\-]+)',
            r'so the answer is\s*([\d,\.\-]+)',
            r'\$\s*([\d,\.\-]+)',
            r'([\d,\.\-]+)\s*dollars?',
            r'([\d,\.\-]+)\s*cents?',
            r'equals?\s*([\d,\.\-]+)',
            r'=\s*([\d,\.\-]+)',
            r'is\s*([\d,\.\-]+)',
            r'solution:\s*([\d,\.\-]+)',
            r'result:\s*([\d,\.\-]+)',
        ]

        response_lower = response.lower()

        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                number = match.group(1).replace(',', '').strip()
                # 验证是否为有效数字
                try:
                    float(number)
                    return number
                except ValueError:
                    continue

        # 如果没有找到模式，提取所有数字并选择最可能的答案
        numbers = re.findall(r'[\d,\.\-]+', response)
        if numbers:
            # 清理数字并验证
            valid_numbers = []
            for num in numbers:
                clean_num = num.replace(',', '').strip()
                try:
                    float(clean_num)
                    valid_numbers.append(clean_num)
                except ValueError:
                    continue

            if valid_numbers:
                # 返回最后一个有效数字
                return valid_numbers[-1]

        return None

    @staticmethod
    def normalize_number(num_str: str) -> Optional[str]:
        """规范化数值字符串"""
        try:
            # 移除逗号并转换为浮点数，再转回字符串以标准化格式
            clean_num = num_str.replace(',', '').strip()
            float_val = float(clean_num)

            # 如果是整数，返回整数格式
            if float_val.is_integer():
                return str(int(float_val))
            else:
                return str(float_val)
        except:
            return None

    @staticmethod
    def numbers_equal(num1: str, num2: str, tolerance: float = 1e-6) -> bool:
        """比较两个数值是否相等（考虑浮点数精度）"""
        try:
            val1 = float(num1.replace(',', ''))
            val2 = float(num2.replace(',', ''))
            return abs(val1 - val2) < tolerance
        except:
            return False

    @staticmethod
    def evaluate_open_ended_qa(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        开放式问答评估（如PopQA）- 修复了过于宽松的评估逻辑
        """
        if len(predictions) != len(references):
            return {'accuracy': 0.0, 'total': 0}

        correct = 0
        total = len(predictions)

        for pred, ref_list in zip(predictions, references):
            pred_clean = pred.lower().strip()

            # 更严格的匹配逻辑
            for ref in ref_list:
                ref_clean = ref.lower().strip()

                # 要求完整单词匹配，而不是简单的子串包含
                if StandardMetricCalculator._word_match(pred_clean, ref_clean):
                    correct += 1
                    break

        accuracy = correct / total if total > 0 else 0.0
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'metric_name': 'word_match_accuracy'
        }

    @staticmethod
    def _word_match(prediction: str, reference: str) -> bool:
        """
        检查预测是否包含完整的参考答案（单词级别匹配）
        """
        import re

        # 移除标点符号并分割单词
        pred_words = set(re.findall(r'\b\w+\b', prediction.lower()))
        ref_words = set(re.findall(r'\b\w+\b', reference.lower()))

        # 检查参考答案的所有单词是否都在预测中
        if len(ref_words) == 0:
            return False

        # 如果参考答案只有一个单词，检查完整匹配
        if len(ref_words) == 1:
            return ref_words.issubset(pred_words)

        # 对于多单词答案，要求所有单词都存在
        return ref_words.issubset(pred_words)


class FewShotPromptGenerator:
    """Few-shot提示生成器，支持可配置的few-shot数量"""

    def __init__(self, few_shot_count: int = 5):
        self.few_shot_count = few_shot_count

    def generate_mmlu_prompt(self, sample: Dict) -> str:
        """生成MMLU标准few-shot提示"""
        # 获取few-shot示例
        few_shot_examples = get_few_shot_prompts('mmlu', self.few_shot_count)

        # 格式化few-shot提示
        few_shot_string = format_few_shot_string('mmlu', few_shot_examples, sample)

        # 添加测试问题
        test_question = f"{sample['formatted_question']}\nAnswer:"

        if few_shot_string:
            return f"{few_shot_string}\n\n{test_question}"
        else:
            return f"The following is a multiple choice question about {sample['subject']}.\n\n{test_question}"

    def generate_gsm8k_prompt(self, sample: Dict) -> str:
        """生成GSM8K标准few-shot提示"""
        few_shot_examples = get_few_shot_prompts('gsm8k', self.few_shot_count)
        few_shot_string = format_few_shot_string('gsm8k', few_shot_examples)

        test_question = f"Question: {sample['question']}\nAnswer:"

        if few_shot_string:
            return f"{few_shot_string}\n\n{test_question}"
        else:
            return test_question

    def generate_math_prompt(self, sample: Dict) -> str:
        """生成MATH竞赛题标准few-shot提示"""
        few_shot_examples = get_few_shot_prompts('math', self.few_shot_count)
        few_shot_string = format_few_shot_string('math', few_shot_examples)

        test_question = f"Problem: {sample['problem']}\nSolution:"

        if few_shot_string:
            return f"{few_shot_string}\n\n{test_question}"
        else:
            return test_question

    def generate_popqa_prompt(self, sample: Dict) -> str:
        """生成PopQA标准few-shot提示"""
        few_shot_examples = get_few_shot_prompts('popqa', self.few_shot_count)
        few_shot_string = format_few_shot_string('popqa', few_shot_examples)

        test_question = f"Question: {sample['question']}\nAnswer:"

        if few_shot_string:
            return f"{few_shot_string}\n\n{test_question}"
        else:
            return test_question

    def generate_alpaca_eval_prompt(self, sample: Dict) -> str:
        """生成AlpacaEval标准few-shot提示"""
        few_shot_examples = get_few_shot_prompts('alpaca_eval2', self.few_shot_count)
        few_shot_string = format_few_shot_string('alpaca_eval2', few_shot_examples)

        test_instruction = f"### Instruction:\n{sample['instruction']}\n\n### Response:"

        if few_shot_string:
            return f"{few_shot_string}\n\n{test_instruction}"
        else:
            return test_instruction

    def generate_ifeval_prompt(self, sample: Dict) -> str:
        """生成IFEval标准few-shot提示"""
        # IFEval通常使用原始提示，不需要few-shot
        return sample['prompt']


class StandardEvaluationPipeline:
    """
    标准评估流水线，增加了详细数据保存和结果分析功能
    """

    def __init__(self, args):
        self.args = args
        self.data_loader = DatasetLoader(args.data_dir)
        self.metric_calculator = StandardMetricCalculator()
        self.prompt_generator = FewShotPromptGenerator(args.few_shot_count)
        self.results = {}
        self.detailed_results = {}  # 新增：保存详细的样本级结果

        # 设置随机种子以确保可重复性
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        print(f"评估配置:")
        print(f"  Few-shot数量: {args.few_shot_count}")
        print(f"  样本数量: {args.num_samples}")
        print(f"  最大生成token数: {args.max_new_tokens}")
        print(f"  批大小: {args.batch_size}")
        print(f"  数据精度: {args.dtype}")
        print(f"  设备映射: {args.device_map}")
        print(f"  随机种子: {args.seed}")

    def evaluate_single_model(self, model_path: str) -> Dict[str, Dict]:
        """评估单个模型在所有数据集上的性能"""
        model_name = os.path.basename(model_path).replace('allenai_', '')
        print(f"\n{'=' * 60}")
        print(f"开始评估模型: {model_name}")
        print(f"{'=' * 60}")

        # 初始化评估器
        evaluator = BatchModelEvaluator(
            model_path,
            self.args.max_new_tokens,
            self.args.batch_size,
            self.args.dtype,
            self.args.device_map
        )

        if not evaluator.load_model():
            return {}

        results = {}
        detailed_results = {}

        # 定义要评估的数据集
        datasets_to_evaluate = [
            ('mmlu', self.evaluate_mmlu),
            ('gsm8k', self.evaluate_gsm8k),
            ('math', self.evaluate_math),
            ('popqa', self.evaluate_popqa),
            ('alpaca_eval2', self.evaluate_alpaca_eval),
            ('ifeval', self.evaluate_ifeval),
        ]

        for dataset_name, eval_method in datasets_to_evaluate:
            try:
                print(f"\n开始评估 {dataset_name.upper()} (样本数: {self.args.num_samples})")
                result, detailed_result = eval_method(evaluator)
                results[dataset_name] = result
                detailed_results[dataset_name] = detailed_result

                # 保存详细生成数据为JSONL格式
                self.save_generation_data_jsonl(model_name, dataset_name, detailed_result)

                print(f"✓ {dataset_name.upper()} 准确率: {result.get('accuracy', 0.0):.3f}")
            except Exception as e:
                print(f"✗ 评估{dataset_name}时出错: {e}")
                import traceback
                traceback.print_exc()
                results[dataset_name] = {'accuracy': 0.0, 'error': str(e)}
                detailed_results[dataset_name] = []

        # 保存模型的详细结果
        self.detailed_results[model_name] = detailed_results

        return results

    def evaluate_mmlu(self, evaluator: BatchModelEvaluator) -> Tuple[Dict, List[Dict]]:
        """评估MMLU数据集，返回聚合结果和详细结果"""
        samples = self.data_loader.load_mmlu(self.args.num_samples)
        if not samples:
            return {'accuracy': 0.0, 'error': 'No samples loaded'}, []

        # 生成提示
        prompts = [self.prompt_generator.generate_mmlu_prompt(sample) for sample in samples]

        # 批量生成响应
        responses = evaluator.batch_generate(prompts)

        # 提取答案并评估
        references = [sample['answer_letter'] for sample in samples]

        # 计算聚合指标
        aggregated_result = self.metric_calculator.evaluate_mmlu(responses, references)

        # 准备详细结果
        detailed_results = []
        for i, (sample, prompt, response, reference) in enumerate(zip(samples, prompts, responses, references)):
            pred_answer = self.metric_calculator.extract_choice_answer(response)
            is_correct = (pred_answer == reference.strip().upper())

            detailed_results.append({
                'sample_id': i,
                'dataset': 'mmlu',
                'subject': sample['subject'],
                'question': sample['question'],
                'choices': sample['choices'],
                'correct_answer': reference,
                'prompt': prompt,
                'response': response,
                'predicted_answer': pred_answer,
                'is_correct': is_correct
            })

        return aggregated_result, detailed_results

    def evaluate_gsm8k(self, evaluator: BatchModelEvaluator) -> Tuple[Dict, List[Dict]]:
        """评估GSM8K数据集"""
        samples = self.data_loader.load_gsm8k(self.args.num_samples)
        if not samples:
            return {'accuracy': 0.0, 'error': 'No samples loaded'}, []

        prompts = [self.prompt_generator.generate_gsm8k_prompt(sample) for sample in samples]
        responses = evaluator.batch_generate(prompts)
        references = [sample['answer'] for sample in samples]

        aggregated_result = self.metric_calculator.evaluate_gsm8k(responses, references)

        # 准备详细结果
        detailed_results = []
        for i, (sample, prompt, response, reference) in enumerate(zip(samples, prompts, responses, references)):
            pred_number = self.metric_calculator.extract_final_number(response)
            ref_number = self.metric_calculator.normalize_number(reference)
            is_correct = (pred_number and ref_number and
                          self.metric_calculator.numbers_equal(pred_number, ref_number))

            detailed_results.append({
                'sample_id': i,
                'dataset': 'gsm8k',
                'question': sample['question'],
                'correct_answer': reference,
                'correct_solution': sample['solution'],
                'prompt': prompt,
                'response': response,
                'predicted_number': pred_number,
                'normalized_reference': ref_number,
                'is_correct': is_correct
            })

        return aggregated_result, detailed_results

    def evaluate_math(self, evaluator: BatchModelEvaluator) -> Tuple[Dict, List[Dict]]:
        """评估MATH数据集 - 修复过于宽松的评估标准"""
        samples = self.data_loader.load_math(self.args.num_samples)
        if not samples:
            return {'accuracy': 0.0, 'error': 'No samples loaded'}, []

        prompts = [self.prompt_generator.generate_math_prompt(sample) for sample in samples]
        responses = evaluator.batch_generate(prompts)

        # 更严格的MATH评估：尝试提取最终答案并进行简单匹配
        correct_count = 0
        detailed_results = []

        for i, (sample, prompt, response) in enumerate(zip(samples, prompts, responses)):
            # 从模型响应中提取答案
            predicted_answer = self._extract_math_answer(response)
            # 从标准解答中提取答案
            gold_answer = self._extract_math_answer(sample['solution'])

            # 比较答案
            is_correct = self._compare_math_answers(predicted_answer, gold_answer)
            if is_correct:
                correct_count += 1

            detailed_results.append({
                'sample_id': i,
                'dataset': 'math',
                'problem': sample['problem'],
                'level': sample['level'],
                'type': sample['type'],
                'correct_solution': sample['solution'],
                'gold_answer': gold_answer,
                'prompt': prompt,
                'response': response,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct
            })

        accuracy = correct_count / len(responses) if responses else 0.0

        aggregated_result = {
            'accuracy': accuracy,
            'correct': correct_count,
            'total': len(responses),
            'metric_name': 'math_exact_match'
        }

        return aggregated_result, detailed_results

    def _extract_math_answer(self, text: str) -> str:
        """从数学文本中提取最终答案"""
        if not text:
            return ""

        # 查找 \\boxed{} 格式的答案
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        boxed_matches = re.findall(boxed_pattern, text)
        if boxed_matches:
            return boxed_matches[-1].strip()

        # 查找其他常见的答案格式
        answer_patterns = [
            r'final answer is\s*([^\n\.]+)',
            r'answer is\s*([^\n\.]+)',
            r'therefore,?\s*([^\n\.]+)',
            r'thus,?\s*([^\n\.]+)',
            r'solution is\s*([^\n\.]+)',
            r'answer:\s*([^\n\.]+)',
        ]

        text_lower = text.lower()
        for pattern in answer_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                return matches[-1].strip()

        # 如果找不到明确的答案标记，返回最后一个数学表达式
        math_expressions = re.findall(r'[=]\s*([^\n\s]+)', text)
        if math_expressions:
            return math_expressions[-1].strip()

        return ""

    def _compare_math_answers(self, pred: str, gold: str) -> bool:
        """比较两个数学答案是否相等"""
        if not pred or not gold:
            return False

        # 清理答案文本
        pred_clean = re.sub(r'[^\w\d\.\-\+\*/\(\)]', '', pred.lower())
        gold_clean = re.sub(r'[^\w\d\.\-\+\*/\(\)]', '', gold.lower())

        # 直接字符串比较
        if pred_clean == gold_clean:
            return True

        # 尝试数值比较
        try:
            pred_num = float(re.findall(r'[\d\.\-]+', pred_clean)[0]) if re.findall(r'[\d\.\-]+', pred_clean) else None
            gold_num = float(re.findall(r'[\d\.\-]+', gold_clean)[0]) if re.findall(r'[\d\.\-]+', gold_clean) else None

            if pred_num is not None and gold_num is not None:
                return abs(pred_num - gold_num) < 1e-6
        except:
            pass

        return False

    def evaluate_popqa(self, evaluator: BatchModelEvaluator) -> Tuple[Dict, List[Dict]]:
        """评估PopQA数据集"""
        samples = self.data_loader.load_popqa(self.args.num_samples)
        if not samples:
            return {'accuracy': 0.0, 'error': 'No samples loaded'}, []

        prompts = [self.prompt_generator.generate_popqa_prompt(sample) for sample in samples]
        responses = evaluator.batch_generate(prompts)
        references = [sample['possible_answers'] for sample in samples]

        aggregated_result = self.metric_calculator.evaluate_open_ended_qa(responses, references)

        # 准备详细结果
        detailed_results = []
        for i, (sample, prompt, response, ref_list) in enumerate(zip(samples, prompts, responses, references)):
            pred_clean = response.lower().strip()
            is_correct = False
            matched_answer = None

            for ref in ref_list:
                if self.metric_calculator._word_match(pred_clean, ref.lower().strip()):
                    is_correct = True
                    matched_answer = ref
                    break

            detailed_results.append({
                'sample_id': i,
                'dataset': 'popqa',
                'question': sample['question'],
                'possible_answers': ref_list,
                'primary_answer': sample['answer'],
                'prompt': prompt,
                'response': response,
                'matched_answer': matched_answer,
                'is_correct': is_correct
            })

        return aggregated_result, detailed_results

    def evaluate_alpaca_eval(self, evaluator: BatchModelEvaluator) -> Tuple[Dict, List[Dict]]:
        """评估AlpacaEval数据集 - 修复评估标准"""
        samples = self.data_loader.load_alpaca_eval(self.args.num_samples)
        if not samples:
            return {'accuracy': 0.0, 'error': 'No samples loaded'}, []

        prompts = [self.prompt_generator.generate_alpaca_eval_prompt(sample) for sample in samples]
        responses = evaluator.batch_generate(prompts)

        # 更合理的AlpacaEval评估：检查响应的帮助性和完整性
        helpful_responses = 0
        detailed_results = []

        for i, (sample, prompt, response) in enumerate(zip(samples, prompts, responses)):
            is_helpful = self._evaluate_helpfulness(response, sample['instruction'])
            if is_helpful:
                helpful_responses += 1

            detailed_results.append({
                'sample_id': i,
                'dataset': 'alpaca_eval2',
                'instruction': sample['instruction'],
                'reference_output': sample.get('output', ''),
                'generator': sample.get('generator', ''),
                'prompt': prompt,
                'response': response,
                'is_helpful': is_helpful,
                'is_correct': is_helpful
            })

        accuracy = helpful_responses / len(responses) if responses else 0.0

        aggregated_result = {
            'accuracy': accuracy,
            'correct': helpful_responses,
            'total': len(responses),
            'metric_name': 'helpfulness_score'
        }

        return aggregated_result, detailed_results

    def _evaluate_helpfulness(self, response: str, instruction: str) -> bool:
        """评估响应的帮助性"""
        if not response or len(response.strip()) < 20:
            return False

        response_lower = response.lower().strip()

        # 检查拒绝回答的模式
        refusal_patterns = [
            "i cannot", "i can't", "i'm unable", "i am unable",
            "i don't know", "i'm not sure", "sorry, i can't",
            "i cannot help", "i'm not able", "i cannot provide"
        ]

        for pattern in refusal_patterns:
            if pattern in response_lower:
                return False

        # 检查是否包含具体信息或建议
        helpful_indicators = [
            "here", "you can", "try", "consider", "suggest", "recommend",
            "step", "way", "method", "approach", "solution", "answer",
            "example", "instance", "such as", "including", "like"
        ]

        helpful_count = sum(1 for indicator in helpful_indicators if indicator in response_lower)

        # 需要至少包含一些帮助性指标，且长度合理
        return helpful_count >= 2 and len(response.strip()) >= 30

    def evaluate_ifeval(self, evaluator: BatchModelEvaluator) -> Tuple[Dict, List[Dict]]:
        """评估IFEval数据集 - 简化评估"""
        samples = self.data_loader.load_ifeval(self.args.num_samples)
        if not samples:
            return {'accuracy': 0.0, 'error': 'No samples loaded'}, []

        prompts = [self.prompt_generator.generate_ifeval_prompt(sample) for sample in samples]
        responses = evaluator.batch_generate(prompts)

        # 简化的指令遵循评估
        valid_responses = 0
        detailed_results = []

        reject_words = ['cannot', 'unable', 'sorry', 'i can\'t', 'i cannot']

        for i, (sample, prompt, response) in enumerate(zip(samples, prompts, responses)):
            is_following = (len(response.strip()) > 5 and
                            not any(reject_word in response.lower() for reject_word in reject_words))
            if is_following:
                valid_responses += 1

            detailed_results.append({
                'sample_id': i,
                'dataset': 'ifeval',
                'prompt': prompt,
                'instruction_id_list': sample['instruction_id_list'],
                'response': response,
                'is_following': is_following,
                'is_correct': is_following
            })

        accuracy = valid_responses / len(responses) if responses else 0.0

        aggregated_result = {
            'accuracy': accuracy,
            'correct': valid_responses,
            'total': len(responses),
            'metric_name': 'instruction_following'
        }

        return aggregated_result, detailed_results

    def run_evaluation(self):
        """运行完整的评估流程"""
        print("开始Few-shot标准化模型评估流程...")

        # 评估所有模型
        for model_path in self.args.model_paths:
            model_name = os.path.basename(model_path).replace('allenai_', '')
            self.results[model_name] = self.evaluate_single_model(model_path)

        # 保存和展示结果
        self.save_results()
        self.save_detailed_results()
        self.generate_latex_table()
        self.print_summary()

    def save_results(self):
        """保存聚合的评估结果"""
        os.makedirs(self.args.output_dir, exist_ok=True)

        # 保存原始结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.args.output_dir, f"evaluation_results_{timestamp}.json")

        # 添加元数据
        results_with_metadata = {
            'metadata': {
                'timestamp': timestamp,
                'args': vars(self.args),
                'total_models': len(self.args.model_paths),
                'few_shot_count': self.args.few_shot_count,
                'num_samples': self.args.num_samples,
                'max_new_tokens': self.args.max_new_tokens,
                'datasets_evaluated': [k for k in self.results.get(list(self.results.keys())[0], {}).keys() if
                                       k != 'error'] if self.results else []
            },
            'results': self.results
        }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_with_metadata, f, ensure_ascii=False, indent=2)

        print(f"✓ 聚合评估结果已保存: {results_file}")

    def save_detailed_results(self):
        """保存详细的样本级评估结果"""
        if not self.detailed_results:
            return

        os.makedirs(self.args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for model_name, model_detailed_results in self.detailed_results.items():
            detailed_file = os.path.join(self.args.output_dir, f"detailed_results_{model_name}_{timestamp}.json")

            detailed_data = {
                'metadata': {
                    'model_name': model_name,
                    'timestamp': timestamp,
                    'evaluation_args': vars(self.args)
                },
                'detailed_results': model_detailed_results
            }

            with open(detailed_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_data, f, ensure_ascii=False, indent=2)

            print(f"✓ {model_name} 详细结果已保存: {detailed_file}")

    def save_generation_data_jsonl(self, model_name: str, dataset_name: str, detailed_results: List[Dict]):
        """
        保存生成数据为JSONL格式，文件名包含所有重要参数
        """
        if not detailed_results:
            return

        os.makedirs(self.args.output_dir, exist_ok=True)

        # 解析模型信息
        model_type = "sft" if "sft" in model_name.lower() else "base"
        model_scale = self._extract_model_scale(model_name)

        # 构建文件名，包含所有重要参数
        filename = (f"generation_data_{dataset_name}_{model_name}_"
                    f"samples{self.args.num_samples}_"
                    f"fewshot{self.args.few_shot_count}_"
                    f"tokens{self.args.max_new_tokens}_"
                    f"scale{model_scale}_{model_type}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

        filepath = os.path.join(self.args.output_dir, filename)

        # 准备保存的数据，包含所有生成信息
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in detailed_results:
                # 构建完整的样本数据
                generation_data = {
                    'metadata': {
                        'model_name': model_name,
                        'model_type': model_type,
                        'model_scale': model_scale,
                        'dataset': dataset_name,
                        'few_shot_count': self.args.few_shot_count,
                        'max_new_tokens': self.args.max_new_tokens,
                        'num_samples': self.args.num_samples,
                        'batch_size': self.args.batch_size,
                        'dtype': self.args.dtype,
                        'timestamp': datetime.now().isoformat()
                    },
                    'sample_id': sample['sample_id'],
                    'input_data': self._extract_input_data(sample),
                    'gold_answer': self._extract_gold_answer(sample),
                    'prompt': sample['prompt'],
                    'generated_text': sample['response'],
                    'generated_tokens': len(sample['response'].split()) if sample['response'] else 0,
                    'generated_chars': len(sample['response']) if sample['response'] else 0,
                    'prediction': self._extract_prediction(sample),
                    'is_correct': sample.get('is_correct', False),
                    'evaluation_details': self._extract_evaluation_details(sample)
                }

                # 写入JSONL格式（每行一个JSON对象）
                f.write(json.dumps(generation_data, ensure_ascii=False) + '\n')

        print(f"✓ {dataset_name} {model_name} 生成数据已保存: {filepath}")

    def _extract_model_scale(self, model_name: str) -> str:
        """从模型名称中提取规模信息"""
        # 查找常见的模型规模标识
        scale_patterns = [
            r'(\d+B)', r'(\d+b)', r'(\d+\.?\d*B)', r'(\d+\.?\d*b)',
            r'(\d+M)', r'(\d+m)', r'(\d+\.?\d*M)', r'(\d+\.?\d*m)'
        ]

        for pattern in scale_patterns:
            match = re.search(pattern, model_name)
            if match:
                return match.group(1).upper()

        # 如果没有找到明确的规模，根据名称推断
        if '1b' in model_name.lower() or '1B' in model_name:
            return '1B'
        elif '7b' in model_name.lower() or '7B' in model_name:
            return '7B'
        elif '13b' in model_name.lower() or '13B' in model_name:
            return '13B'
        elif '32b' in model_name.lower() or '32B' in model_name:
            return '32B'

        return 'unknown'

    def _extract_input_data(self, sample: Dict) -> Dict:
        """提取输入数据"""
        dataset = sample['dataset']
        if dataset == 'mmlu':
            return {
                'question': sample['question'],
                'choices': sample['choices'],
                'subject': sample['subject']
            }
        elif dataset == 'gsm8k':
            return {
                'question': sample['question']
            }
        elif dataset == 'math':
            return {
                'problem': sample['problem'],
                'level': sample['level'],
                'type': sample['type']
            }
        elif dataset == 'popqa':
            return {
                'question': sample['question']
            }
        elif dataset == 'alpaca_eval2':
            return {
                'instruction': sample['instruction']
            }
        elif dataset == 'ifeval':
            return {
                'prompt': sample.get('original_prompt', sample['prompt']),
                'instruction_id_list': sample.get('instruction_id_list', [])
            }
        else:
            return {}

    def _extract_gold_answer(self, sample: Dict) -> str:
        """提取标准答案"""
        dataset = sample['dataset']
        if dataset == 'mmlu':
            return sample['correct_answer']
        elif dataset == 'gsm8k':
            return sample['correct_answer']
        elif dataset == 'math':
            return sample.get('gold_answer', '')
        elif dataset == 'popqa':
            return sample.get('primary_answer', sample.get('possible_answers', [''])[0])
        elif dataset == 'alpaca_eval2':
            return sample.get('reference_output', '')
        elif dataset == 'ifeval':
            return 'N/A'  # IFEval没有标准答案
        else:
            return ''

    def _extract_prediction(self, sample: Dict) -> str:
        """提取模型预测"""
        dataset = sample['dataset']
        if dataset == 'mmlu':
            return sample.get('predicted_answer', '')
        elif dataset == 'gsm8k':
            return sample.get('predicted_number', '')
        elif dataset == 'math':
            return sample.get('predicted_answer', '')
        elif dataset == 'popqa':
            return sample.get('matched_answer', '')
        elif dataset == 'alpaca_eval2':
            return 'helpful' if sample.get('is_helpful', False) else 'not_helpful'
        elif dataset == 'ifeval':
            return 'following' if sample.get('is_following', False) else 'not_following'
        else:
            return ''

    def _extract_evaluation_details(self, sample: Dict) -> Dict:
        """提取评估细节"""
        dataset = sample['dataset']
        details = {'dataset': dataset}

        if dataset == 'mmlu':
            details.update({
                'subject': sample['subject'],
                'predicted_answer': sample.get('predicted_answer', ''),
                'correct_answer': sample['correct_answer']
            })
        elif dataset == 'gsm8k':
            details.update({
                'predicted_number': sample.get('predicted_number', ''),
                'normalized_reference': sample.get('normalized_reference', ''),
                'extraction_successful': bool(sample.get('predicted_number'))
            })
        elif dataset == 'math':
            details.update({
                'level': sample['level'],
                'type': sample['type'],
                'predicted_answer': sample.get('predicted_answer', ''),
                'gold_answer': sample.get('gold_answer', '')
            })
        elif dataset == 'popqa':
            details.update({
                'possible_answers': sample.get('possible_answers', []),
                'matched_answer': sample.get('matched_answer', ''),
                'exact_match': bool(sample.get('matched_answer'))
            })
        elif dataset == 'alpaca_eval2':
            details.update({
                'is_helpful': sample.get('is_helpful', False),
                'response_length': len(sample['response']) if sample['response'] else 0
            })
        elif dataset == 'ifeval':
            details.update({
                'is_following': sample.get('is_following', False),
                'instruction_id_list': sample.get('instruction_id_list', [])
            })

        return details

    def generate_latex_table(self):
        """生成LaTeX格式的结果表格"""
        if not self.results:
            print("没有评估结果可以生成表格")
            return

        # 数据集名称映射
        dataset_names = {
            'ifeval': 'IFEval',
            'alpaca_eval2': 'AlpacaEval2',
            'mmlu': 'MMLU',
            'popqa': 'PopQA',
            'gsm8k': 'GSM8K',
            'math': 'MATH'
        }

        # 创建DataFrame
        df_data = []
        for model_name, scores in self.results.items():
            row = {'Model': model_name}
            for dataset_key, dataset_display in dataset_names.items():
                accuracy = scores.get(dataset_key, {}).get('accuracy', 0.0)
                row[dataset_display] = f"{accuracy:.3f}"
            df_data.append(row)

        df = pd.DataFrame(df_data)

        # 生成LaTeX表格
        latex_table = df.to_latex(index=False, escape=False, float_format="%.3f")

        # 保存LaTeX表格
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        latex_file = os.path.join(self.args.output_dir, f"results_table_{timestamp}.tex")
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_table)

        print(f"✓ LaTeX表格已保存: {latex_file}")
        print("\nLaTeX表格内容:")
        print(latex_table)

    def print_summary(self):
        """打印评估结果摘要"""
        print(f"\n{'=' * 80}")
        print(f"Few-shot评估结果摘要 (few-shot数量: {self.args.few_shot_count})")
        print(f"Token生成数量: {self.args.max_new_tokens}, 批大小: {self.args.batch_size}")
        print(f"{'=' * 80}")

        for model_name, scores in self.results.items():
            print(f"\n模型: {model_name}")
            print("-" * 40)
            for dataset, result in scores.items():
                if isinstance(result, dict) and 'accuracy' in result:
                    accuracy = result['accuracy']
                    total = result.get('total', 'N/A')
                    correct = result.get('correct', 'N/A')
                    print(f"  {dataset.upper():12} : {accuracy:.3f} ({correct}/{total})")
                elif isinstance(result, dict) and 'error' in result:
                    print(f"  {dataset.upper():12} : ERROR - {result['error']}")


def main():
    """主函数，解析参数并运行Few-shot标准化评估"""
    parser = argparse.ArgumentParser(description="Fixed Few-shot标准化OLMo-2模型下游任务评估框架")

    # 模型相关参数
    parser.add_argument("--model_paths", nargs="+",
                        default=[
                            "/root/autodl-tmp/ift_memorization/model_cache/allenai_OLMo-2-0425-1B",
                            # "/root/autodl-tmp/ift_memorization/model_cache/allenai_OLMo-2-0425-1B-SFT",
                            # "/root/autodl-tmp/ift_memorization/model_cache/allenai_OLMo-2-1124-7B",
                            # "/root/autodl-tmp/ift_memorization/model_cache/allenai_OLMo-2-1124-7B-SFT",
                            # "/root/autodl-tmp/ift_memorization/model_cache/allenai_OLMo-2-1124-13B",
                            # "/root/autodl-tmp/ift_memorization/model_cache/allenai_OLMo-2-1124-13B-SFT",
                            # "/root/autodl-tmp/ift_memorization/model_cache/allenai_OLMo-2-0325-32B",
                            # "/root/autodl-tmp/ift_memorization/model_cache/allenai_OLMo-2-0325-32B-SFT",
                        ], help="要评估的模型路径列表")

    # 数据相关参数
    parser.add_argument("--data_dir", default="../../data/downstream_test_data",
                        help="测试数据集保存目录")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="每个数据集的样本数量")
    parser.add_argument("--few_shot_count", type=int, default=5,
                        help="Few-shot示例数量")

    # 生成参数 - 修复了默认值
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="最大生成token数量 (增加以支持数学推理)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批处理大小 (减小以避免OOM)")
    parser.add_argument("--dtype", default="float32",
                        choices=["float16", "float32", "bfloat16"],
                        help="模型数据精度 (修改默认为float32)")
    parser.add_argument("--device_map", default="cuda",
                        help="设备映射策略 (修改默认为cuda)")

    # 输出参数
    parser.add_argument("--output_dir", default="/root/autodl-tmp/ift_memorization/results/exp2_fixed",
                        help="评估结果输出目录")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()

    # 打印修复说明
    print("=" * 80)
    print("Few-shot标准化评估框架 - 修复版本")
    print("=" * 80)
    print("修复内容:")
    print("1. ✓ 默认设备映射改为cuda，精度改为float32")
    print("2. ✓ 增加详细数据保存功能（输入输出和指标）")
    print("3. ✓ 修复OOM问题，添加动态批大小调整")
    print("4. ✓ 修复GSM8K评估问题，增加max_new_tokens到256")
    print("5. ✓ 修复PopQA评估过于宽松的问题")
    print("6. ✓ 增强数值提取和选择题答案提取")
    print("7. ✓ 添加内存管理和错误处理")
    print("=" * 80)

    # 创建并运行评估流水线
    pipeline = StandardEvaluationPipeline(args)
    pipeline.run_evaluation()


if __name__ == "__main__":
    main()