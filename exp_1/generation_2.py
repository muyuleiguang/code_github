#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验1.1 - 模型生成内容（修改版）
使用base和sft模型对预训练测试数据进行推理生成，保存完整的生成信息用于后续分析
增加动态批处理功能
"""

import json
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import time
import gc


def load_model_and_tokenizer(model_path: str, device: str = 'auto') -> Tuple[Any, Any]:
    """
    加载模型和分词器

    Args:
        model_path: 模型路径
        device: 设备类型 ('auto', 'cuda', 'cpu')

    Returns:
        model: 加载的模型
        tokenizer: 对应的分词器
    """
    print(f"正在加载模型: {model_path}")

    # 设置设备
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32 if device == 'cuda' else torch.float32,
        trust_remote_code=True,
        device_map='auto'  # if device == 'cuda' else None
    )

    if device != 'cuda':
        model = model.to(device)

    model.eval()

    print(f"模型加载完成, 设备: {model.device}")
    return model, tokenizer


def load_test_data(data_dir: str, datasets: List[str], max_samples: int = None) -> Dict[str, List[Dict]]:
    """
    加载测试数据

    Args:
        data_dir: 数据目录
        datasets: 要加载的数据集列表
        max_samples: 每种类型最大样本数，None表示加载全部

    Returns:
        data_dict: 按类型分组的测试数据
    """
    data_dict = {}

    file_mapping = {
        'stackexchange': 'stackexchange_instruction.jsonl',
        'dclm-privacy': 'dclm_privacy.jsonl',
        'wiki-fact': 'wiki_fact.jsonl'
    }

    for data_type in datasets:
        if data_type not in file_mapping:
            print(f"警告: 未知数据集类型 {data_type}")
            continue

        filename = file_mapping[data_type]
        filepath = os.path.join(data_dir, filename)

        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data_list = [json.loads(line) for line in f]

            # 限制样本数量
            if max_samples and len(data_list) > max_samples:
                data_list = data_list[:max_samples]

            data_dict[data_type] = data_list
            print(f"加载 {data_type}: {len(data_list)} 条样本")
        else:
            print(f"警告: 文件不存在 {filepath}")
            data_dict[data_type] = []

    return data_dict


def extract_text_from_sample(sample: Dict) -> str:
    """
    从样本中提取文本内容

    Args:
        sample: 数据样本

    Returns:
        text: 提取的文本内容
    """
    # 尝试不同的文本字段名
    text_fields = ['text', 'content', 'passage', 'document']

    for field in text_fields:
        if field in sample and isinstance(sample[field], str):
            return sample[field]

    # 如果没有找到标准字段，寻找最长的字符串字段
    max_len = 0
    max_text = ""
    for key, value in sample.items():
        if isinstance(value, str) and len(value) > max_len:
            max_len = len(value)
            max_text = value

    return max_text


def prepare_batch_inputs(batch_samples: List[Dict], prefix_length: int, max_new_tokens: int, tokenizer: Any) -> Tuple[
    List[Dict], torch.Tensor, List[int]]:
    """
    为批处理准备输入数据

    Args:
        batch_samples: 批处理样本列表
        prefix_length: 前缀长度
        max_new_tokens: 最大生成token数
        tokenizer: 分词器

    Returns:
        batch_info: 每个样本的处理信息
        input_ids: 批处理的输入tensor
        prefix_lengths: 每个样本的实际前缀长度
    """
    batch_info = []
    input_ids_list = []
    prefix_lengths = []

    for sample in batch_samples:
        text = extract_text_from_sample(sample)
        if not text or len(text.strip()) < 50:
            continue

        # 对原始文本进行tokenization
        full_tokens = tokenizer.encode(text, add_special_tokens=False)

        # 检查前缀长度是否合理
        actual_prefix_length = prefix_length
        if prefix_length >= len(full_tokens):
            actual_prefix_length = max(1, len(full_tokens) - 10)

        # 提取前缀和原始续写
        prefix_tokens = full_tokens[:actual_prefix_length]
        original_continuation_tokens = full_tokens[actual_prefix_length:actual_prefix_length + max_new_tokens]

        # 保存样本信息
        batch_info.append({
            'sample': sample,
            'text': text,
            'full_tokens': full_tokens,
            'prefix_tokens': prefix_tokens,
            'original_continuation_tokens': original_continuation_tokens,
            'prefix_text': tokenizer.decode(prefix_tokens, skip_special_tokens=True),
            'original_continuation': tokenizer.decode(original_continuation_tokens, skip_special_tokens=True)
        })

        input_ids_list.append(prefix_tokens)
        prefix_lengths.append(len(prefix_tokens))

    if not input_ids_list:
        return [], torch.empty((0, 0)), []

    # 创建批处理tensor，使用padding
    max_prefix_len = max(len(ids) for ids in input_ids_list)
    batch_input_ids = []

    for ids in input_ids_list:
        padded_ids = ids + [tokenizer.pad_token_id] * (max_prefix_len - len(ids))
        batch_input_ids.append(padded_ids)

    input_ids = torch.tensor(batch_input_ids)

    return batch_info, input_ids, prefix_lengths


def generate_batch_with_prefix(model: Any, tokenizer: Any, batch_samples: List[Dict],
                               prefix_length: int, max_new_tokens: int = 100,
                               temperature: float = 0.0, top_k_save: int = 10) -> List[Dict[str, Any]]:
    """
    批处理生成文本，并保存详细信息

    Args:
        model: 语言模型
        tokenizer: 分词器
        batch_samples: 批处理样本列表
        prefix_length: 前缀长度（token数）
        max_new_tokens: 最大生成token数
        temperature: 生成温度，0表示贪婪解码
        top_k_save: 保存top-k的概率信息

    Returns:
        results: 包含生成结果和分析信息的字典列表
    """
    with torch.no_grad():
        # 准备批处理输入
        batch_info, input_ids, prefix_lengths = prepare_batch_inputs(
            batch_samples, prefix_length, max_new_tokens, tokenizer
        )

        if not batch_info:
            return []

        # 移动到设备
        input_ids = input_ids.to(model.device)

        # 创建attention mask
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        # 生成文本
        generation_config = {
            'max_new_tokens': max_new_tokens,
            'do_sample': temperature > 0,
            'temperature': temperature if temperature > 0 else 1.0,
            'pad_token_id': tokenizer.pad_token_id,
            'attention_mask': attention_mask,
            'return_dict_in_generate': True,
            'output_scores': True,  # 返回每步的logits
        }

        outputs = model.generate(input_ids, **generation_config)

        # 处理每个样本的结果
        results = []
        for i, info in enumerate(batch_info):
            try:
                # 获取原始前缀长度（去除padding）
                original_prefix_len = prefix_lengths[i]

                # 提取生成的token（不包括输入部分）
                generated_sequence = outputs.sequences[i]
                # 找到实际前缀结束的位置（考虑padding）
                prefix_end_pos = original_prefix_len
                generated_token_ids = generated_sequence[prefix_end_pos:].cpu().tolist()

                # 移除pad tokens
                generated_token_ids = [tid for tid in generated_token_ids if tid != tokenizer.pad_token_id]
                generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

                # 处理logits和概率（只保留top-k）- 简化处理，因为批处理时scores处理比较复杂
                top_tokens_list = []
                if hasattr(outputs, 'scores') and outputs.scores and len(generated_token_ids) > 0:
                    # 由于批处理的复杂性，这里简化处理scores
                    # 如果需要详细的token概率，可以考虑单独处理或使用其他方法
                    for step_idx, step_logits in enumerate(outputs.scores):
                        if step_idx >= len(generated_token_ids):
                            break

                        step_logits_i = step_logits[i].cpu()
                        step_probs = torch.softmax(step_logits_i, dim=-1)

                        # 保存top-k tokens信息
                        top_probs, top_indices = torch.topk(step_probs, k=min(top_k_save, len(step_probs)))
                        top_tokens = [
                            {
                                'token_id': idx.item(),
                                'token_text': tokenizer.decode([idx.item()]),
                                'probability': prob.item(),
                                'rank': rank + 1
                            }
                            for rank, (idx, prob) in enumerate(zip(top_indices, top_probs))
                        ]
                        top_tokens_list.append(top_tokens)

                # 构建结果字典
                result = {
                    'prefix_text': info['prefix_text'],
                    'generated_text': generated_text,
                    'original_continuation': info['original_continuation'],
                    'prefix_tokens': info['prefix_tokens'],
                    'generated_tokens': generated_token_ids,
                    'original_continuation_tokens': info['original_continuation_tokens'],
                    'top_tokens': top_tokens_list,
                }

                results.append(result)

            except Exception as e:
                print(f"处理批次中第 {i} 个样本时出错: {e}")
                # 添加空结果以保持索引对应
                results.append({
                    'prefix_text': "",
                    'generated_text': "",
                    'original_continuation': "",
                    'prefix_tokens': [],
                    'generated_tokens': [],
                    'original_continuation_tokens': [],
                    'top_tokens': [],
                })
                continue

        return results


class DynamicBatchProcessor:
    """动态批处理器"""

    def __init__(self, initial_batch_size: int = 128, min_batch_size: int = 1, max_batch_size: int = 256):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.success_count = 0
        self.consecutive_success_threshold = 5  # 连续成功几次后尝试增加批处理大小

    def adjust_batch_size_on_oom(self):
        """OOM时减少批处理大小"""
        old_size = self.current_batch_size
        self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
        self.success_count = 0
        print(f"OOM检测到，批处理大小从 {old_size} 减少到 {self.current_batch_size}")

    def adjust_batch_size_on_success(self):
        """成功时可能增加批处理大小"""
        self.success_count += 1
        if (self.success_count >= self.consecutive_success_threshold and
                self.current_batch_size < self.max_batch_size):
            old_size = self.current_batch_size
            self.current_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.5))
            self.success_count = 0
            print(
                f"连续成功 {self.consecutive_success_threshold} 次，批处理大小从 {old_size} 增加到 {self.current_batch_size}")

    def get_batch_size(self) -> int:
        return self.current_batch_size


def process_single_dataset(model: Any, tokenizer: Any, data_type: str, data_list: List[Dict],
                           prefix_length: int, max_new_tokens: int, top_k_save: int) -> List[Dict]:
    """
    处理单个数据集在特定前缀长度下的生成（使用动态批处理）

    Args:
        model: 语言模型
        tokenizer: 分词器
        data_type: 数据集类型
        data_list: 数据样本列表
        prefix_length: 前缀长度
        max_new_tokens: 最大生成token数
        top_k_save: 保存top-k概率信息

    Returns:
        results: 生成结果列表
    """
    results = []
    batch_processor = DynamicBatchProcessor(initial_batch_size=512, max_batch_size=1024)

    # 过滤有效样本
    valid_samples = []
    for i, sample in enumerate(data_list):
        text = extract_text_from_sample(sample)
        if text and len(text.strip()) >= 50:
            sample_with_id = sample.copy()
            sample_with_id['sample_id'] = i
            valid_samples.append(sample_with_id)

    if not valid_samples:
        print(f"数据集 {data_type} 没有有效样本")
        return results

    # 使用tqdm显示进度
    pbar = tqdm(total=len(valid_samples), desc=f"生成-{data_type}-prefix{prefix_length}-new{max_new_tokens}")

    i = 0
    while i < len(valid_samples):
        batch_size = batch_processor.get_batch_size()
        batch_samples = valid_samples[i:i + batch_size]

        try:
            # 批处理生成
            batch_results = generate_batch_with_prefix(
                model, tokenizer, batch_samples, prefix_length,
                max_new_tokens, top_k_save=top_k_save
            )

            # 添加元信息并合并结果
            for j, (sample, result) in enumerate(zip(batch_samples, batch_results)):
                result.update({
                    'sample_id': sample.get('sample_id', i + j),
                    'original_sample': {k: v for k, v in sample.items() if k != 'sample_id'},
                })
                results.append(result)

            # 成功处理，调整批处理大小
            batch_processor.adjust_batch_size_on_success()

            # 更新进度
            pbar.update(len(batch_samples))
            i += len(batch_samples)

            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                print(f"批处理大小 {batch_size} 时发生OOM")
                batch_processor.adjust_batch_size_on_oom()

                # 清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # 如果批处理大小已经是最小值，仍然OOM，则跳过这些样本
                if batch_processor.get_batch_size() == batch_processor.min_batch_size:
                    print(
                        f"即使使用最小批处理大小 {batch_processor.min_batch_size} 仍然OOM，跳过 {len(batch_samples)} 个样本")
                    pbar.update(len(batch_samples))
                    i += len(batch_samples)

                # 不增加i，重新尝试当前批次（使用更小的批处理大小）
            else:
                print(f"处理批次时出现非OOM错误: {e}")
                # 跳过当前批次
                pbar.update(len(batch_samples))
                i += len(batch_samples)

        except Exception as e:
            print(f"处理批次时出错: {e}")
            # 跳过当前批次
            pbar.update(len(batch_samples))
            i += len(batch_samples)

    pbar.close()
    return results


def save_results_jsonl(results: List[Dict], data_type: str, prefix_length: int, max_new_tokens: int,
                       model_name: str, model_type: str, output_dir: str, num_samples: int):
    """
    保存生成结果为jsonl格式

    Args:
        results: 生成结果列表
        data_type: 数据集类型
        prefix_length: 前缀长度
        model_name: 模型名称
        model_type: 模型类型
        output_dir: 输出目录
        num_samples: 样本数量
    """
    os.makedirs(output_dir, exist_ok=True)

    # 生成文件名：dataset_prefix{length}_{model_name}_{model_type}_{num_samples}samples.jsonl
    filename = f"{data_type}_prefix{prefix_length}_new{max_new_tokens}_{model_name}_{model_type}_{num_samples}_samples.jsonl"
    filepath = os.path.join(output_dir, filename)

    # 保存结果
    with open(filepath, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"结果已保存到: {filepath} ({len(results)} 条记录)")


def main():
    parser = argparse.ArgumentParser(description='使用base和sft模型生成测试内容')

    # 模型相关参数
    parser.add_argument('--model_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/model_cache',
                        help='模型根目录')
    parser.add_argument('--model_name', type=str, default='allenai_OLMo-2-1124-13B',
                        choices=['allenai_OLMo-2-0425-1B', 'allenai_OLMo-2-1124-7B',
                                 'allenai_OLMo-2-1124-13B', 'allenai_OLMo-2-0325-32B'],
                        help='模型名称')
    parser.add_argument('--model_type', type=str, default='sft',
                        choices=['base', 'sft'],
                        help='模型类型：base或sft')

    # 数据相关参数
    parser.add_argument('--data_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/data/pretraining_test_data/mem_test',
                        help='测试数据目录')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['stackexchange'],
                        choices=['stackexchange', 'dclm-privacy', 'wiki-fact'],
                        help='要测试的数据集列表')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='每种数据类型的最大样本数，None表示全部')

    # 生成相关参数
    parser.add_argument('--prefix_lengths', type=int, nargs='+', default=[64],
                        help='前缀长度列表')
    parser.add_argument('--max_new_tokens', type=int, nargs='+', default=[8, 16],
                        help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='生成温度，0表示贪婪解码')
    parser.add_argument('--top_k_save', type=int, default=10,
                        help='保存top-k的概率信息')

    # 输出相关参数
    parser.add_argument('--output_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/results/exp1_generation_',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='使用的设备')

    args = parser.parse_args()


    # 构建完整模型路径
    if args.model_type == 'sft':
        model_path = os.path.join(args.model_dir, f"{args.model_name}-SFT")
    else:
        model_path = os.path.join(args.model_dir, args.model_name)

    print("开始生成实验...")
    print(f"模型路径: {model_path}")
    print(f"模型类型: {args.model_type}")
    print(f"数据目录: {args.data_dir}")
    print(f"数据集: {args.datasets}")
    print(f"前缀长度: {args.prefix_lengths}")
    print(f"最大样本数: {args.max_samples}")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(model_path, args.device)

    # 加载测试数据
    data_dict = load_test_data(args.data_dir, args.datasets, args.max_samples)

    if not any(data_dict.values()):
        print("错误: 未加载到任何有效数据")
        return

    # 提取模型规模信息
    model_scale = args.model_name.split('-')[-1]  # 提取规模信息 (如 "1B")

    # 先遍历前缀长度，后遍历数据集
    start_time = time.time()
    total_processed = 0

    for prefix_length in args.prefix_lengths:
        print(f"\n处理前缀长度: {prefix_length}")

        for data_type, data_list in data_dict.items():
            if not data_list:
                print(f"跳过空数据集: {data_type}")
                continue

            print(f"  处理数据集: {data_type} ({len(data_list)} 个样本)")

            # 处理单个数据集
            for max_tokens in args.max_new_tokens:
                print(f"  生成tokens数目: {max_tokens}")
                output_dir = args.output_dir + str(max_tokens)
                results = process_single_dataset(
                    model, tokenizer, data_type, data_list,
                    prefix_length, max_tokens, args.top_k_save
                )

                # 保存结果
                save_results_jsonl(
                    results, data_type, prefix_length, max_tokens,
                    model_scale, args.model_type, output_dir, len(results)
                )

                total_processed += len(results)

    end_time = time.time()
    print(f"\n实验完成! 总共处理 {total_processed} 条记录，耗时: {end_time - start_time:.2f}秒")


if __name__ == "__main__":
    main()