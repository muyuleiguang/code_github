#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 1.1 - Model-generated content (revised)
Use base and sft models to run inference on pretraining test data and save full generation info for subsequent analysis
Add dynamic batching functionality
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
    Load model and tokenizer

    Args:
        model_path: Model path
        device: Device type ('auto', 'cuda', 'cpu')

    Returns:
        model: Loaded model
        tokenizer: Corresponding tokenizer
    """
    print(f"正在加载模型: {model_path}")

    # Set device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
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
    Load test data

    Args:
        data_dir: Data directory
        datasets: List of datasets to load
        max_samples: Maximum number of samples per dataset; None means load all

    Returns:
        data_dict: Test data grouped by type
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

            # Limit the number of samples
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
    Extract text content from a sample

    Args:
        sample: Data sample

    Returns:
        text: Extracted text content
    """
    # Try different possible text field names
    text_fields = ['text', 'content', 'passage', 'document']

    for field in text_fields:
        if field in sample and isinstance(sample[field], str):
            return sample[field]

    # If no standard field is found, choose the longest string field
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
    Prepare input data for batching

    Args:
        batch_samples: List of samples in the batch
        prefix_length: Prefix length
        max_new_tokens: Maximum number of new tokens to generate
        tokenizer: Tokenizer

    Returns:
        batch_info: Processing info for each sample
        input_ids: Batched input tensor
        prefix_lengths: Actual prefix length for each sample
    """
    batch_info = []
    input_ids_list = []
    prefix_lengths = []

    for sample in batch_samples:
        text = extract_text_from_sample(sample)
        if not text or len(text.strip()) < 50:
            continue

        # Tokenize the raw text
        full_tokens = tokenizer.encode(text, add_special_tokens=False)

        # Check whether prefix length is reasonable
        actual_prefix_length = prefix_length
        if prefix_length >= len(full_tokens):
            actual_prefix_length = max(1, len(full_tokens) - 10)

        # Extract prefix and the original continuation
        prefix_tokens = full_tokens[:actual_prefix_length]
        original_continuation_tokens = full_tokens[actual_prefix_length:actual_prefix_length + max_new_tokens]

        # Save sample info
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

    # Create batched tensor with padding
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
    Generate text in batch and save detailed information

    Args:
        model: Language model
        tokenizer: Tokenizer
        batch_samples: List of samples in the batch
        prefix_length: Prefix length (number of tokens)
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Generation temperature; 0 means greedy decoding
        top_k_save: Save top-k probability info

    Returns:
        results: List of dicts containing generation results and analysis info
    """
    with torch.no_grad():
        # Prepare batched inputs
        batch_info, input_ids, prefix_lengths = prepare_batch_inputs(
            batch_samples, prefix_length, max_new_tokens, tokenizer
        )

        if not batch_info:
            return []

        # Move to device
        input_ids = input_ids.to(model.device)

        # Create attention mask
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        # Generate text
        generation_config = {
            'max_new_tokens': max_new_tokens,
            'do_sample': temperature > 0,
            'temperature': temperature if temperature > 0 else 1.0,
            'pad_token_id': tokenizer.pad_token_id,
            'attention_mask': attention_mask,
            'return_dict_in_generate': True,
            'output_scores': True,  # Return logits at each step
        }

        outputs = model.generate(input_ids, **generation_config)

        # Process results for each sample
        results = []
        for i, info in enumerate(batch_info):
            try:
                # Original prefix length (without padding)
                original_prefix_len = prefix_lengths[i]

                # Extract generated tokens (excluding the input portion)
                generated_sequence = outputs.sequences[i]
                # Locate the end position of the actual prefix (considering padding)
                prefix_end_pos = original_prefix_len
                generated_token_ids = generated_sequence[prefix_end_pos:].cpu().tolist()

                # Remove pad tokens
                generated_token_ids = [tid for tid in generated_token_ids if tid != tokenizer.pad_token_id]
                generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

                # Process logits/probabilities (keep top-k only) - simplified since batch scores are more complex
                top_tokens_list = []
                if hasattr(outputs, 'scores') and outputs.scores and len(generated_token_ids) > 0:
                    # Due to batching complexity, handle scores in a simplified manner here
                    # If detailed token probabilities are needed, consider single-sample processing or other methods
                    for step_idx, step_logits in enumerate(outputs.scores):
                        if step_idx >= len(generated_token_ids):
                            break

                        step_logits_i = step_logits[i].cpu()
                        step_probs = torch.softmax(step_logits_i, dim=-1)

                        # Save top-k token info
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

                # Build result dict
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
                # Add an empty result to preserve index alignment
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
    """Dynamic batch processor"""

    def __init__(self, initial_batch_size: int = 128, min_batch_size: int = 1, max_batch_size: int = 256):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.success_count = 0
        self.consecutive_success_threshold = 5  # Increase batch size after N consecutive successes

    def adjust_batch_size_on_oom(self):
        """Reduce batch size on OOM"""
        old_size = self.current_batch_size
        self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
        self.success_count = 0
        print(f"OOM检测到，批处理大小从 {old_size} 减少到 {self.current_batch_size}")

    def adjust_batch_size_on_success(self):
        """Optionally increase batch size on success"""
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
    Process generation for a single dataset under a specific prefix length (using dynamic batching)

    Args:
        model: Language model
        tokenizer: Tokenizer
        data_type: Dataset type
        data_list: List of data samples
        prefix_length: Prefix length
        max_new_tokens: Maximum number of new tokens to generate
        top_k_save: Save top-k probability info

    Returns:
        results: List of generation results
    """
    results = []
    batch_processor = DynamicBatchProcessor(initial_batch_size=512, max_batch_size=1024)

    # Filter valid samples
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

    # Use tqdm to show progress
    pbar = tqdm(total=len(valid_samples), desc=f"生成-{data_type}-prefix{prefix_length}-new{max_new_tokens}")

    i = 0
    while i < len(valid_samples):
        batch_size = batch_processor.get_batch_size()
        batch_samples = valid_samples[i:i + batch_size]

        try:
            # Batch generation
            batch_results = generate_batch_with_prefix(
                model, tokenizer, batch_samples, prefix_length,
                max_new_tokens, top_k_save=top_k_save
            )

            # Add metadata and merge results
            for j, (sample, result) in enumerate(zip(batch_samples, batch_results)):
                result.update({
                    'sample_id': sample.get('sample_id', i + j),
                    'original_sample': {k: v for k, v in sample.items() if k != 'sample_id'},
                })
                results.append(result)

            # Successfully processed; adjust batch size
            batch_processor.adjust_batch_size_on_success()

            # Update progress
            pbar.update(len(batch_samples))
            i += len(batch_samples)

            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                print(f"批处理大小 {batch_size} 时发生OOM")
                batch_processor.adjust_batch_size_on_oom()

                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # If still OOM at the minimum batch size, skip these samples
                if batch_processor.get_batch_size() == batch_processor.min_batch_size:
                    print(
                        f"即使使用最小批处理大小 {batch_processor.min_batch_size} 仍然OOM，跳过 {len(batch_samples)} 个样本")
                    pbar.update(len(batch_samples))
                    i += len(batch_samples)

                # Do not increment i here; retry current batch with a smaller batch size
            else:
                print(f"处理批次时出现非OOM错误: {e}")
                # Skip current batch
                pbar.update(len(batch_samples))
                i += len(batch_samples)

        except Exception as e:
            print(f"处理批次时出错: {e}")
            # Skip current batch
            pbar.update(len(batch_samples))
            i += len(batch_samples)

    pbar.close()
    return results


def save_results_jsonl(results: List[Dict], data_type: str, prefix_length: int, max_new_tokens: int,
                       model_name: str, model_type: str, output_dir: str, num_samples: int):
    """
    Save generation results in JSONL format

    Args:
        results: List of generation results
        data_type: Dataset type
        prefix_length: Prefix length
        model_name: Model name
        model_type: Model type
        output_dir: Output directory
        num_samples: Number of samples
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename: dataset_prefix{length}_new{max_new_tokens}_{model_name}_{model_type}_{num_samples}_samples.jsonl
    filename = f"{data_type}_prefix{prefix_length}_new{max_new_tokens}_{model_name}_{model_type}_{num_samples}_samples.jsonl"
    filepath = os.path.join(output_dir, filename)

    # Save results
    with open(filepath, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"结果已保存到: {filepath} ({len(results)} 条记录)")


def main():
    parser = argparse.ArgumentParser(description='使用base和sft模型生成测试内容')

    # Model-related arguments
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

    # Data-related arguments
    parser.add_argument('--data_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/data/pretraining_test_data/mem_test',
                        help='测试数据目录')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['stackexchange'],
                        choices=['stackexchange', 'dclm-privacy', 'wiki-fact'],
                        help='要测试的数据集列表')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='每种数据类型的最大样本数，None表示全部')

    # Generation-related arguments
    parser.add_argument('--prefix_lengths', type=int, nargs='+', default=[64],
                        help='前缀长度列表')
    parser.add_argument('--max_new_tokens', type=int, nargs='+', default=[8, 16],
                        help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='生成温度，0表示贪婪解码')
    parser.add_argument('--top_k_save', type=int, default=10,
                        help='保存top-k的概率信息')

    # Output-related arguments
    parser.add_argument('--output_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/results/exp1_generation_',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='使用的设备')

    args = parser.parse_args()


    # Build the full model path
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

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path, args.device)

    # Load test data
    data_dict = load_test_data(args.data_dir, args.datasets, args.max_samples)

    if not any(data_dict.values()):
        print("错误: 未加载到任何有效数据")
        return

    # Extract model scale info
    model_scale = args.model_name.split('-')[-1]  # Extract scale info (e.g., "1B")

    # Iterate prefix lengths first, then datasets
    start_time = time.time()
    total_processed = 0

    for prefix_length in args.prefix_lengths:
        print(f"\n处理前缀长度: {prefix_length}")

        for data_type, data_list in data_dict.items():
            if not data_list:
                print(f"跳过空数据集: {data_type}")
                continue

            print(f"  处理数据集: {data_type} ({len(data_list)} 个样本)")

            # Process one dataset
            for max_tokens in args.max_new_tokens:
                print(f"  生成tokens数目: {max_tokens}")
                output_dir = args.output_dir + str(max_tokens)
                results = process_single_dataset(
                    model, tokenizer, data_type, data_list,
                    prefix_length, max_tokens, args.top_k_save
                )

                # Save results
                save_results_jsonl(
                    results, data_type, prefix_length, max_tokens,
                    model_scale, args.model_type, output_dir, len(results)
                )

                total_processed += len(results)

    end_time = time.time()
    print(f"\n实验完成! 总共处理 {total_processed} 条记录，耗时: {end_time - start_time:.2f}秒")


if __name__ == "__main__":
    main()
