#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
英语俚语和中文唐诗记忆（背诵）测试代码 - 修改版
支持batch处理和解决JSON序列化问题
测试base和sft模型在两种数据集上的记忆情况
输入是诗词的上一句/俚语的前n-1个词，输出是下一句/最后一个词

修改内容：
1. 分为两类样本数：num_samples用于测试记忆情况，num_samples_rep_save用于保存表征
2. 生成两个文件：一个不包含表征和logits，一个包含表征和logits
"""

import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import random





def convert_numpy_to_python(obj):
    """
    递归地将numpy数组转换为Python原生类型，以便JSON序列化
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    else:
        return obj


def load_dataset(file_path, dataset_type):
    """
    加载数据集

    Args:
        file_path: 数据集文件路径
        dataset_type: 数据集类型 ('idiom' 或 'poems')

    Returns:
        list: 加载的数据列表
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"警告: 文件 {file_path} 不存在，使用默认示例数据")
        # 使用默认示例数据
        if dataset_type == 'idiom':
            data = [
                {"idiom": "break the ice"},
                {"idiom": "piece of cake"},
                {"idiom": "hit the nail on the head"},
                {"idiom": "barking up the wrong tree"},
                {"idiom": "let the cat out of the bag"}
            ]
        else:  # poems
            data = [
                {"id": 1, "third_row": "春眠不觉晓", "fourth_row": "处处闻啼鸟"},
                {"id": 2, "third_row": "白日依山尽", "fourth_row": "黄河入海流"},
                {"id": 3, "third_row": "举头望明月", "fourth_row": "低头思故乡"},
                {"id": 4, "third_row": "床前明月光", "fourth_row": "疑是地上霜"},
                {"id": 5, "third_row": "野火烧不尽", "fourth_row": "春风吹又生"}
            ]

    print(f"成功加载 {dataset_type} 数据集，共 {len(data)} 条数据")
    return data


def prepare_idiom_prompt(idioms, target_idiom, few_shots):
    """
    准备英语俚语的few-shot prompt

    Args:
        idioms: 俚语数据列表
        target_idiom: 目标俚语
        few_shots: few-shot示例数量

    Returns:
        tuple: (input_text, target_output, input_words, target_word)
    """
    # 使用后few_shots个作为示例（不包括目标俚语）
    examples = idioms[-few_shots:]

    # 构建few-shot prompt
    prompt_parts = ["Complete the following English idioms:\n\n"]

    for example in examples:
        idiom_words = example['idiom'].split()
        if len(idiom_words) > 1:
            input_part = ' '.join(idiom_words[:-1])
            output_part = idiom_words[-1]
            prompt_parts.append(f"Input: {input_part}\nOutput: {output_part}\n\n")

    # 添加目标俚语的输入部分
    target_words = target_idiom['idiom'].split()
    if len(target_words) > 1:
        target_input = ' '.join(target_words[:-1])
        target_output = target_words[-1]
    else:
        # 如果俚语只有一个词，则使用前几个字符作为输入
        target_input = target_idiom['idiom'][:-1]
        target_output = target_idiom['idiom'][-1]

    prompt_parts.append(f"Input: {target_input}\nOutput:")

    input_text = ''.join(prompt_parts)

    return input_text, target_output, target_input, target_output


def prepare_poem_prompt(poems, target_poem, few_shots):
    """
    准备中文唐诗的few-shot prompt

    Args:
        poems: 诗词数据列表
        target_poem: 目标诗词
        few_shots: few-shot示例数量

    Returns:
        tuple: (input_text, target_output, input_line, target_line)
    """
    # 使用后few_shots个作为示例（不包括目标诗词）
    examples = poems[-few_shots:]

    # 构建few-shot prompt
    prompt_parts = ["根据上句诗，补全下句七言绝句，用中文回复：\n\n"]

    for example in examples:
        input_line = example['third_row']
        output_line = example['fourth_row']
        prompt_parts.append(f"上句：{input_line}\n下句：{output_line}\n\n")

    # 添加目标诗词的输入部分
    target_input = target_poem['third_row']
    target_output = target_poem['fourth_row']

    prompt_parts.append(f"上句：{target_input}\n下句：")

    input_text = ''.join(prompt_parts)

    return input_text, target_output, target_input, target_output


def prepare_batch_inputs(input_texts, tokenizer, device, max_length=512):
    """
    准备batch输入

    Args:
        input_texts: 输入文本列表
        tokenizer: 分词器
        device: 设备
        max_length: 最大长度

    Returns:
        dict: batch输入
    """
    # 批量编码
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )

    # 移动到设备
    inputs = {k: v.to(device) for k, v in inputs.items()}

    return inputs


def generate_batch_simple(model, tokenizer, input_texts, max_new_tokens, device):
    """
    批量生成文本（不保存表征，节省内存）

    Args:
        model: 语言模型
        tokenizer: 分词器
        input_texts: 输入文本列表
        max_new_tokens: 最大生成token数
        device: 设备

    Returns:
        list: 每个样本的结果列表，只包含生成文本
    """
    batch_size = len(input_texts)

    # 准备batch输入
    inputs = prepare_batch_inputs(input_texts, tokenizer, device)
    input_lengths = inputs['attention_mask'].sum(dim=1).cpu().numpy()

    # 存储所有样本的结果
    batch_results = []

    # 初始化每个样本的结果存储
    for i in range(batch_size):
        batch_results.append({
            'generated_text': '',
            'generated_tokens': [],
            'input_length': input_lengths[i]
        })

    current_input_ids = inputs['input_ids']
    current_attention_mask = inputs['attention_mask']

    with torch.no_grad():
        for step in range(max_new_tokens):
            # 前向传播
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                return_dict=True
            )

            # 获取每个样本最后一个位置的logits
            next_token_logits = outputs.logits[:, -1, :]

            # 贪婪选择下一个token
            next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # 为每个样本保存token信息
            for i in range(batch_size):
                token_id = next_token_ids[i, 0].item()
                batch_results[i]['generated_tokens'].append(token_id)

            # 更新输入序列
            current_input_ids = torch.cat([current_input_ids, next_token_ids], dim=1)

            # 更新attention mask
            new_attention = torch.ones(batch_size, 1, device=device)
            current_attention_mask = torch.cat([current_attention_mask, new_attention], dim=1)

    # 解码生成的文本
    for i, sample_result in enumerate(batch_results):
        generated_tokens = sample_result['generated_tokens']
        sample_result['generated_text'] = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return batch_results


def generate_batch_with_representations(model, tokenizer, input_texts, max_new_tokens, device):
    """
    批量生成文本并提取隐藏层表征

    Args:
        model: 语言模型
        tokenizer: 分词器
        input_texts: 输入文本列表
        max_new_tokens: 最大生成token数
        device: 设备

    Returns:
        list: 每个样本的结果列表，包含生成文本、隐藏状态等
    """
    batch_size = len(input_texts)

    # 准备batch输入
    inputs = prepare_batch_inputs(input_texts, tokenizer, device)
    input_lengths = inputs['attention_mask'].sum(dim=1).cpu().numpy()

    # 存储所有样本的结果
    batch_results = []

    # 初始化每个样本的结果存储
    for i in range(batch_size):
        batch_results.append({
            'generated_text': '',
            'all_hidden_states': [],  # [num_tokens, num_layers, hidden_size]
            'token_logits': [],  # [num_tokens, vocab_size]
            'token_probs': [],  # [num_tokens, vocab_size]
            'generated_tokens': [],  # [num_tokens]
            'input_length': input_lengths[i]
        })

    current_input_ids = inputs['input_ids']
    current_attention_mask = inputs['attention_mask']

    with torch.no_grad():
        for step in range(max_new_tokens):
            # 前向传播
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            # 获取每个样本最后一个位置的logits
            next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)

            # 贪婪选择下一个token
            next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch_size, 1]

            # 为每个样本保存信息
            for i in range(batch_size):
                sample_result = batch_results[i]

                # 保存token信息
                token_id = next_token_ids[i, 0].item()
                sample_result['generated_tokens'].append(token_id)
                sample_result['token_logits'].append(next_token_logits[i].cpu().numpy())
                sample_result['token_probs'].append(next_token_probs[i].cpu().numpy())

                # 保存所有层的隐藏状态（最后一个位置）
                step_hidden_states = []
                for layer_idx, layer_hidden in enumerate(outputs.hidden_states):
                    # layer_hidden: [batch_size, seq_len, hidden_size]
                    last_token_hidden = layer_hidden[i, -1, :].cpu().numpy()  # [hidden_size]
                    step_hidden_states.append(last_token_hidden)
                sample_result['all_hidden_states'].append(step_hidden_states)

            # 更新输入序列
            current_input_ids = torch.cat([current_input_ids, next_token_ids], dim=1)

            # 更新attention mask
            new_attention = torch.ones(batch_size, 1, device=device)
            current_attention_mask = torch.cat([current_attention_mask, new_attention], dim=1)

    # 解码生成的文本
    for i, sample_result in enumerate(batch_results):
        generated_tokens = sample_result['generated_tokens']
        sample_result['generated_text'] = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return batch_results


def process_dataset_batch(model, tokenizer, dataset, dataset_type, args, model_name, model_type):
    """
    批量处理数据集，分为两个阶段：普通测试和表征保存

    Args:
        model: 语言模型
        tokenizer: 分词器
        dataset: 数据集
        dataset_type: 数据集类型 ('idiom' 或 'poems')
        args: 命令行参数
        model_name: 当前模型名称
        model_type: 当前模型类型

    Returns:
        tuple: (普通结果列表, 带表征结果列表)
    """
    # 验证参数
    if args.num_samples_rep_save > args.num_samples:
        print(f"警告: num_samples_rep_save ({args.num_samples_rep_save}) 大于 num_samples ({args.num_samples})")
        print(f"将 num_samples_rep_save 设置为 {args.num_samples}")
        args.num_samples_rep_save = args.num_samples

    # 取样本
    test_samples = dataset[:args.num_samples]
    rep_samples = dataset[:args.num_samples_rep_save]

    print(f"开始处理 {dataset_type} 数据集")
    print(f"普通测试样本数: {len(test_samples)}")
    print(f"表征保存样本数: {len(rep_samples)}")

    # 阶段1: 普通测试（不保存表征）
    print(f"\n阶段1: 普通测试 {dataset_type} 数据集...")
    normal_results = []

    for batch_start in tqdm(range(0, len(test_samples), args.batch_size), desc=f"普通测试{dataset_type}"):
        batch_end = min(batch_start + args.batch_size, len(test_samples))
        batch_samples = test_samples[batch_start:batch_end]

        try:
            # 准备batch的输入
            batch_input_texts = []
            batch_metadata = []

            for i, sample in enumerate(batch_samples):
                sample_id = batch_start + i

                if dataset_type == 'idiom':
                    input_text, target_output, input_part, expected_output = prepare_idiom_prompt(
                        dataset, sample, args.few_shots
                    )
                else:  # poems
                    input_text, target_output, input_part, expected_output = prepare_poem_prompt(
                        dataset, sample, args.few_shots
                    )

                batch_input_texts.append(input_text)
                batch_metadata.append({
                    'sample_id': sample_id,
                    'input_text': input_text,
                    'input_part': input_part,
                    'expected_output': expected_output,
                    'target_output': target_output
                })

            # 批量生成（简单模式）
            batch_results = generate_batch_simple(
                model, tokenizer, batch_input_texts, args.max_new_tokens, args.device
            )

            # 整合结果
            for metadata, generation_result in zip(batch_metadata, batch_results):
                # 清理生成的文本
                generated_text = generation_result['generated_text'].strip().split('\n')[0]

                # 创建完整的结果记录
                result = {
                    'sample_id': metadata['sample_id'],
                    'dataset_type': dataset_type,
                    # 'input_text': metadata['input_text'],
                    'input_part': metadata['input_part'],
                    # 'expected_output': metadata['expected_output'],
                    'generated_text': generated_text,
                    'target_output': metadata['target_output'],
                    'model_type': model_type,
                    'model_name': model_name,
                    'generated_tokens': generation_result['generated_tokens'],
                    # 'num_generated_tokens': len(generation_result['generated_tokens']),
                    # 'input_length': int(generation_result['input_length']),
                    # 'has_representations': False
                }

                normal_results.append(result)

        except Exception as e:
            print(f"处理普通测试batch {batch_start}-{batch_end} 时出错: {e}")
            continue

    # 阶段2: 表征保存测试
    print(f"\n阶段2: 表征保存 {dataset_type} 数据集...")
    rep_results = []

    for batch_start in tqdm(range(0, len(rep_samples), args.batch_size), desc=f"表征保存{dataset_type}"):
        batch_end = min(batch_start + args.batch_size, len(rep_samples))
        batch_samples = rep_samples[batch_start:batch_end]

        try:
            # 准备batch的输入
            batch_input_texts = []
            batch_metadata = []

            for i, sample in enumerate(batch_samples):
                sample_id = batch_start + i

                if dataset_type == 'idiom':
                    input_text, target_output, input_part, expected_output = prepare_idiom_prompt(
                        dataset, sample, args.few_shots
                    )
                else:  # poems
                    input_text, target_output, input_part, expected_output = prepare_poem_prompt(
                        dataset, sample, args.few_shots
                    )

                batch_input_texts.append(input_text)
                batch_metadata.append({
                    'sample_id': sample_id,
                    'input_text': input_text,
                    'input_part': input_part,
                    'expected_output': expected_output,
                    'target_output': target_output
                })

            # 批量生成（带表征）
            batch_results = generate_batch_with_representations(
                model, tokenizer, batch_input_texts, args.max_new_tokens, args.device
            )

            # 整合结果
            for metadata, generation_result in zip(batch_metadata, batch_results):
                # 清理生成的文本
                generated_text = generation_result['generated_text'].strip().split('\n')[0]

                # 创建完整的结果记录
                result = {
                    'sample_id': metadata['sample_id'],
                    'dataset_type': dataset_type,
                    # 'input_text': metadata['input_text'],
                    'input_part': metadata['input_part'],
                    # 'expected_output': metadata['expected_output'],
                    'generated_text': generated_text,
                    'target_output': metadata['target_output'],
                    'model_type': model_type,
                    'model_name': model_name,
                    'generated_tokens': generation_result['generated_tokens'],
                    'num_layers': len(generation_result['all_hidden_states'][0]) if generation_result[
                        'all_hidden_states'] else 0,
                    'num_generated_tokens': len(generation_result['generated_tokens']),
                    'input_length': int(generation_result['input_length']),
                    'has_representations': True,
                    # 转换numpy数组为Python列表以支持JSON序列化
                    'all_hidden_states': convert_numpy_to_python(generation_result['all_hidden_states']),
                    'token_logits': convert_numpy_to_python(generation_result['token_logits']),
                    'token_probs': convert_numpy_to_python(generation_result['token_probs'])
                }

                rep_results.append(result)

        except Exception as e:
            print(f"处理表征保存batch {batch_start}-{batch_end} 时出错: {e}")
            continue

    return normal_results, rep_results


def save_results(results, args, model_name, model_type, result_type="normal"):
    """
    保存结果到文件

    Args:
        results: 结果列表
        args: 命令行参数
        model_name: 模型名称
        model_type: 模型类型
        result_type: 结果类型 ("normal" 或 "representations")
    """
    os.makedirs(args.save_dir, exist_ok=True)

    # 生成文件名
    model_name_clean = model_name.split('-')[-1]
    if result_type == "representations":
        filename = f"idiom_{model_name_clean}_{model_type}_with_representations.json"
    else:
        filename = f"idiom_{model_name_clean}_{model_type}.json"

    save_path = os.path.join(args.save_dir, filename)

    # 准备保存数据
    save_data = {
        'args': vars(args),
        'model_name': model_name,
        'model_type': model_type,
        'result_type': result_type,
        'results': convert_numpy_to_python(results),
        'total_samples': len(results),
        'dataset_types': list(set([r['dataset_type'] for r in results]))
    }

    # 保存结果
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到: {save_path}")


def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description='测试模型在俚语和诗词上的记忆能力')

    # 数据集相关参数
    parser.add_argument('--data_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/data/test_data_2',
                        help='数据集目录路径')
    parser.add_argument('--idiom_file', type=str, default='idiom.jsonl',
                        help='英语俚语数据集文件名')
    parser.add_argument('--poems_file', type=str, default='poems.jsonl',
                        help='中文唐诗数据集文件名')

    # 模型相关参数
    parser.add_argument('--model_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/model_cache',
                        help='模型根目录')
    parser.add_argument('--model_names', type=str, nargs='+',
                        default=['allenai_OLMo-2-1124-13B'],
                        choices=['allenai_OLMo-2-0425-1B', 'allenai_OLMo-2-1124-7B',
                                 'allenai_OLMo-2-1124-13B', 'allenai_OLMo-2-0325-32B'],
                        help='模型名称列表')
    parser.add_argument('--model_type', type=str, nargs='+',
                        default=['base', 'sft'],
                        choices=['base', 'sft'],
                        help='模型类型列表：base和/或sft')
    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备')

    # 生成参数
    parser.add_argument('--max_new_tokens', type=int, default=7,
                        help='最大生成token数（俚语和诗词都是7）')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch处理大小')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='测试记忆情况的样本数量（取前num_samples个）')
    parser.add_argument('--num_samples_rep_save', type=int, default=10,
                        help='保存表征的样本数量（取前num_samples_rep_save个，应小于num_samples）')
    parser.add_argument('--few_shots', type=int, default=5,
                        help='few-shot prompt中的示例数量（取后few_shots个）')

    # 输出参数
    parser.add_argument('--save_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/results/exp2_memorization_poem',
                        help='结果保存目录')
    parser.add_argument('--save_generations', action='store_true', default=True,
                        help='是否保存生成结果到文件')

    return parser.parse_args()

def main():
    """主函数"""
    args = setup_args()

    print(f"开始测试模型: {args.model_names}")
    print(f"模型类型: {args.model_type}")
    print(f"设备: {args.device}")
    print(f"批处理大小: {args.batch_size}")
    print(f"普通测试样本数: {args.num_samples}")
    print(f"表征保存样本数: {args.num_samples_rep_save}")
    print(f"最大生成token数: {args.max_new_tokens}")

    # 设置随机种子
    random.seed(42)
    torch.manual_seed(42)

    # 加载数据集
    idiom_path = os.path.join(args.data_dir, args.idiom_file)
    poems_path = os.path.join(args.data_dir, args.poems_file)

    idiom_data = load_dataset(idiom_path, 'idiom')
    poems_data = load_dataset(poems_path, 'poems')

    # 迭代所有模型名称和类型的组合
    for model_name in args.model_names:
        for model_type in args.model_type:
            print(f"\n{'=' * 60}")
            print(f"处理模型: {model_name} ({model_type})")
            print(f"{'=' * 60}")

            # 构造实际的模型路径
            if model_type == 'sft':
                actual_model_name = os.path.join(args.model_dir, f"{model_name}-SFT")
            else:
                actual_model_name = os.path.join(args.model_dir, model_name)
            print(f"实际模型路径: {actual_model_name}")

            # 加载模型和分词器
            print("加载模型和分词器...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(actual_model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    actual_model_name,
                    torch_dtype=torch.float32,
                    device_map='auto' if args.device == 'cuda' else None
                )

                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                print("模型和分词器加载完成")

            except Exception as e:
                print(f"加载模型失败: {e}")
                print("跳过当前模型...")
                continue

            all_normal_results = []
            all_rep_results = []

            # 处理俚语数据集
            if idiom_data:
                idiom_normal, idiom_rep = process_dataset_batch(
                    model, tokenizer, idiom_data, 'idiom', args, model_name, model_type
                )
                all_normal_results.extend(idiom_normal)
                all_rep_results.extend(idiom_rep)

            # 处理诗词数据集
            if poems_data:
                poems_normal, poems_rep = process_dataset_batch(
                    model, tokenizer, poems_data, 'poems', args, model_name, model_type
                )
                all_normal_results.extend(poems_normal)
                all_rep_results.extend(poems_rep)

            # 保存结果
            if args.save_generations:
                if all_normal_results:
                    save_results(all_normal_results, args, model_name, model_type, "normal")
                if all_rep_results:
                    save_results(all_rep_results, args, model_name, model_type, "representations")

            print(f"模型 {model_name} ({model_type}) 测试完成！")
            print(f"普通测试样本: {len(all_normal_results)}")
            print(f"表征保存样本: {len(all_rep_results)}")
            print(f"俚语普通样本: {len([r for r in all_normal_results if r['dataset_type'] == 'idiom'])}")
            print(f"诗词普通样本: {len([r for r in all_normal_results if r['dataset_type'] == 'poems'])}")
            print(f"俚语表征样本: {len([r for r in all_rep_results if r['dataset_type'] == 'idiom'])}")
            print(f"诗词表征样本: {len([r for r in all_rep_results if r['dataset_type'] == 'poems'])}")

            # 清理GPU内存
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n所有模型测试完成！")
    print(f"结果保存目录: {args.save_dir}")


if __name__ == "__main__":
    main()