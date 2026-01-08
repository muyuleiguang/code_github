"""
Memorization evaluation metrics implementation
Implement 5 core memorization measurement metrics
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import editdistance
import warnings

warnings.filterwarnings('ignore')


class MemorizationMetrics:
    def __init__(
            self,
            tokenizer_name: str = None,
            sentence_model_name: str = "/root/autodl-tmp/ift_memorization/model_cache/sentence_transformers"
    ):
        """
        Initialize evaluation metrics.

        Args:
            tokenizer_name: tokenizer model name (optional)
            sentence_model_name: sentence embedding model name/path
        """
        self.tokenizer = None
        if tokenizer_name:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except:
                print(f"警告: 无法加载tokenizer {tokenizer_name}")

        try:
            self.sentence_model = SentenceTransformer(sentence_model_name)
        except Exception as e:
            print(f"警告: 加载 sentence model '{sentence_model_name}' 时发生错误。")
            # print(f"Specific error type: {type(e).__name__}")
            # print(f"Error details: {e}")
            # Print full traceback; useful for debugging
            # traceback.print_exc()
            self.sentence_model = None

        self.rouge = Rouge()
        self.smoothing = SmoothingFunction()

    def exact_match_rate(
            self,
            generated_tokens: List[List[int]],
            reference_tokens: List[List[int]]
    ) -> Dict[str, float]:
        """
        Method 1: Exact Match Rate (Exact Match Rate)

        The proportion of samples where the model output exactly matches the training
        reference text. This is the strictest metric.

        Args:
            generated_tokens: list of generated token sequences
            reference_tokens: list of reference token sequences

        Returns:
            exact match metrics
        """
        assert len(generated_tokens) == len(reference_tokens)

        exact_matches = 0
        for gen_tokens, ref_tokens in zip(generated_tokens, reference_tokens):
            if gen_tokens == ref_tokens:
                exact_matches += 1

        return {
            "exact_match_rate": exact_matches / len(generated_tokens),
            "exact_matches": exact_matches,
            "total_samples": len(generated_tokens)
        }

    def rouge_bleu_scores(
            self,
            generated_texts: List[str],
            reference_texts: List[str],
            generated_tokens: List[List[int]] = None,
            reference_tokens: List[List[int]] = None
    ) -> Dict[str, float]:
        """
        Method 2: ROUGE / BLEU scores

        Measures n-gram overlap between generated text and reference text.
        Can capture near-verbatim memorization.

        Args:
            generated_texts: list of generated texts
            reference_texts: list of reference texts
            generated_tokens: list of generated token sequences (optional, for token-level BLEU)
            reference_tokens: list of reference token sequences (optional, for token-level BLEU)

        Returns:
            ROUGE and BLEU metrics
        """
        results = {}

        # ROUGE scores (text-based)
        if generated_texts and reference_texts:
            assert len(generated_texts) == len(reference_texts)

            rouge_1_scores = []
            rouge_2_scores = []
            rouge_l_scores = []

            for gen, ref in zip(generated_texts, reference_texts):
                if not gen or not ref:
                    rouge_1_scores.append(0.0)
                    rouge_2_scores.append(0.0)
                    rouge_l_scores.append(0.0)
                    continue

                try:
                    scores = self.rouge.get_scores(gen, ref)[0]
                    rouge_1_scores.append(scores['rouge-1']['f'])
                    rouge_2_scores.append(scores['rouge-2']['f'])
                    rouge_l_scores.append(scores['rouge-l']['f'])
                except:
                    rouge_1_scores.append(0.0)
                    rouge_2_scores.append(0.0)
                    rouge_l_scores.append(0.0)

            results.update({
                "rouge_1_f": np.mean(rouge_1_scores),
                "rouge_2_f": np.mean(rouge_2_scores),
                "rouge_l_f": np.mean(rouge_l_scores),
                "rouge_1_std": np.std(rouge_1_scores),
                "rouge_2_std": np.std(rouge_2_scores),
                "rouge_l_std": np.std(rouge_l_scores)
            })

        # BLEU scores (token-based, more precise)
        if generated_tokens and reference_tokens:
            assert len(generated_tokens) == len(reference_tokens)

            bleu_1_scores = []
            bleu_2_scores = []
            bleu_4_scores = []

            for gen_tokens, ref_tokens in zip(generated_tokens, reference_tokens):
                if not gen_tokens or not ref_tokens:
                    bleu_1_scores.append(0.0)
                    bleu_2_scores.append(0.0)
                    bleu_4_scores.append(0.0)
                    continue

                # Convert token IDs to strings for BLEU computation
                gen_str_tokens = [str(t) for t in gen_tokens]
                ref_str_tokens = [str(t) for t in ref_tokens]

                # BLEU-1
                bleu_1 = sentence_bleu([ref_str_tokens], gen_str_tokens,
                                       weights=(1, 0, 0, 0),
                                       smoothing_function=self.smoothing.method1)
                bleu_1_scores.append(bleu_1)

                # BLEU-2
                bleu_2 = sentence_bleu([ref_str_tokens], gen_str_tokens,
                                       weights=(0.5, 0.5, 0, 0),
                                       smoothing_function=self.smoothing.method1)
                bleu_2_scores.append(bleu_2)

                # BLEU-4
                bleu_4 = sentence_bleu([ref_str_tokens], gen_str_tokens,
                                       weights=(0.25, 0.25, 0.25, 0.25),
                                       smoothing_function=self.smoothing.method1)
                bleu_4_scores.append(bleu_4)

            results.update({
                "bleu_1": np.mean(bleu_1_scores),
                "bleu_2": np.mean(bleu_2_scores),
                "bleu_4": np.mean(bleu_4_scores),
                "bleu_1_std": np.std(bleu_1_scores),
                "bleu_2_std": np.std(bleu_2_scores),
                "bleu_4_std": np.std(bleu_4_scores)
            })

        return results

    def edit_distance_metrics(
            self,
            generated_tokens: List[List[int]],
            reference_tokens: List[List[int]],
            generated_texts: List[str] = None,
            reference_texts: List[str] = None
    ) -> Dict[str, float]:
        """
        Method 3: Edit distance (Edit Distance)

        How many insertions, deletions, and substitutions are required to transform the
        generated output into the reference. Smaller distance indicates stronger memorization.
        Primarily uses token-level edit distance.

        Args:
            generated_tokens: list of generated token sequences
            reference_tokens: list of reference token sequences
            generated_texts: list of generated texts (optional)
            reference_texts: list of reference texts (optional)

        Returns:
            edit distance metrics
        """
        assert len(generated_tokens) == len(reference_tokens)

        token_distances = []
        normalized_token_distances = []

        # Token-level edit distance (primary metric)
        for gen_tokens, ref_tokens in zip(generated_tokens, reference_tokens):
            token_dist = editdistance.eval(gen_tokens, ref_tokens)
            token_distances.append(token_dist)

            # Normalized token edit distance
            max_token_len = max(len(gen_tokens), len(ref_tokens))
            if max_token_len > 0:
                normalized_token_distances.append(token_dist / max_token_len)
            else:
                normalized_token_distances.append(0.0)

        result = {
            "token_edit_distance": np.mean(token_distances),
            "normalized_token_distance": np.mean(normalized_token_distances),
            "token_distance_std": np.std(token_distances),
            "normalized_token_distance_std": np.std(normalized_token_distances)
        }

        # Character-level edit distance (auxiliary metric)
        if generated_texts and reference_texts:
            char_distances = []
            normalized_char_distances = []

            for i, (gen_text, ref_text) in enumerate(zip(generated_texts, reference_texts)):
                if i < len(generated_tokens):  # ensure index alignment
                    char_dist = editdistance.eval(gen_text, ref_text)
                    char_distances.append(char_dist)

                    max_char_len = max(len(gen_text), len(ref_text))
                    if max_char_len > 0:
                        normalized_char_distances.append(char_dist / max_char_len)
                    else:
                        normalized_char_distances.append(0.0)

            if char_distances:
                result.update({
                    "char_edit_distance": np.mean(char_distances),
                    "normalized_char_distance": np.mean(normalized_char_distances),
                    "char_distance_std": np.std(char_distances),
                    "normalized_char_distance_std": np.std(normalized_char_distances)
                })

        return result

    def semantic_similarity(
            self,
            generated_texts: List[str],
            reference_texts: List[str]
    ) -> Dict[str, float]:
        """
        Method 4: Semantic similarity

        Compute embedding similarity using SentenceBERT or similar models.

        Args:
            generated_texts: list of generated texts
            reference_texts: list of reference texts

        Returns:
            semantic similarity metrics
        """
        if self.sentence_model is None:
            return {
                "cosine_similarity": 0.0,
                "similarity_std": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "median_similarity": 0.0
            }

        assert len(generated_texts) == len(reference_texts)

        try:
            gen_embeddings = self.sentence_model.encode(generated_texts)
            ref_embeddings = self.sentence_model.encode(reference_texts)

            similarities = []
            for gen_emb, ref_emb in zip(gen_embeddings, ref_embeddings):
                cosine_sim = np.dot(gen_emb, ref_emb) / (np.linalg.norm(gen_emb) * np.linalg.norm(ref_emb))
                similarities.append(cosine_sim)

            return {
                "cosine_similarity": np.mean(similarities),
                "similarity_std": np.std(similarities),
                "min_similarity": np.min(similarities),
                "max_similarity": np.max(similarities),
                "median_similarity": np.median(similarities)
            }
        except Exception as e:
            print(f"计算语义相似度时出错: {e}")
            return {
                "cosine_similarity": 0.0,
                "similarity_std": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "median_similarity": 0.0
            }

    def likelihood_ppl_loss_metrics(
            self,
            top_tokens_list: List[List[List[Dict]]],
            reference_tokens: List[List[int]],
            logits: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Method 5: Likelihood, PPL, loss, logits

        Compute memorization-related metrics based on model output probabilities.

        Args:
            top_tokens_list: Top-k token probability information for each step of each sample
                Format: [sample][step][top_k_tokens]
                Each token_info contains: {'token_id': int, 'probability': float, 'rank': int}
            reference_tokens: reference token sequences
            logits: full logits tensors (optional; more precise if available)

        Returns:
            likelihood, perplexity, loss related metrics
        """
        if not top_tokens_list or not reference_tokens:
            return {
                "avg_log_likelihood": float('-inf'),
                "perplexity": float('inf'),
                "avg_loss": float('inf'),
                "target_token_probability": 0.0,
                "target_token_rank": float('inf'),
                "target_in_top1_rate": 0.0,
                "target_in_top3_rate": 0.0,
                "target_in_top5_rate": 0.0
            }

        log_likelihoods = []
        losses = []
        target_probs = []
        target_ranks = []
        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        total_positions = 0

        for sample_idx, (sample_top_tokens, ref_tokens) in enumerate(zip(top_tokens_list, reference_tokens)):
            if not sample_top_tokens or not ref_tokens:
                continue

            sample_log_likelihood = 0.0
            sample_positions = 0

            for step_idx, step_top_tokens in enumerate(sample_top_tokens):
                if step_idx >= len(ref_tokens):
                    break

                target_token = ref_tokens[step_idx]
                total_positions += 1
                sample_positions += 1

                # Find the target token's rank and probability in top-k
                target_found = False

                for rank, token_info in enumerate(step_top_tokens):
                    token_id = token_info.get('token_id')
                    prob = token_info.get('probability', 0.0)

                    if token_id == target_token:
                        target_probs.append(prob)
                        target_ranks.append(rank + 1)  # rank starts from 1
                        target_found = True

                        # Compute log likelihood
                        if prob > 0:
                            log_prob = np.log(prob)
                            sample_log_likelihood += log_prob

                            # Compute cross-entropy loss
                            loss = -log_prob
                            losses.append(loss)
                        else:
                            sample_log_likelihood += -100  # avoid log(0)
                            losses.append(100)

                        # Top-k hit statistics
                        if rank == 0:  # top-1
                            top1_hits += 1
                        if rank < 3:   # top-5
                            top3_hits += 1
                        if rank < 5:  # top-10
                            top5_hits += 1
                        break

                if not target_found:
                    # If the target token is not in top-k, use a very small probability
                    target_probs.append(1e-10)
                    target_ranks.append(float('inf'))
                    sample_log_likelihood += -100
                    losses.append(100)

            if sample_positions > 0:
                log_likelihoods.append(sample_log_likelihood / sample_positions)

        # Compute aggregate metrics
        avg_log_likelihood = np.mean(log_likelihoods) if log_likelihoods else float('-inf')
        perplexity = np.exp(-avg_log_likelihood) if avg_log_likelihood != float('-inf') else float('inf')
        avg_loss = np.mean(losses) if losses else float('inf')
        avg_target_prob = np.mean(target_probs) if target_probs else 0.0

        finite_ranks = [r for r in target_ranks if r != float('inf')]
        avg_target_rank = np.mean(finite_ranks) if finite_ranks else float('inf')

        return {
            "avg_log_likelihood": float(avg_log_likelihood),
            "perplexity": float(perplexity),
            "avg_loss": float(avg_loss),
            "target_token_probability": float(avg_target_prob),
            "target_token_rank": float(avg_target_rank),
            "target_in_top1_rate": top1_hits / total_positions if total_positions > 0 else 0.0,
            "target_in_top3_rate": top3_hits / total_positions if total_positions > 0 else 0.0,
            "target_in_top5_rate": top5_hits / total_positions if total_positions > 0 else 0.0,
            "target_prob_std": float(np.std(target_probs)) if target_probs else 0.0,
            "total_positions": total_positions
        }

    def compute_all_metrics_from_data(
            self,
            samples: List[Dict]
    ) -> Dict[str, Dict]:
        """
        Compute all 5 evaluation metric families from sample data.

        Args:
            samples: sample list; each sample contains fields such as generated_tokens,
                     original_continuation_tokens, etc.

        Returns:
            dictionary containing results for all metrics
        """
        # Extract data
        generated_tokens = []
        reference_tokens = []
        generated_texts = []
        reference_texts = []
        top_tokens_list = []

        for sample in samples:
            if 'generated_tokens' in sample and 'original_continuation_tokens' in sample:
                generated_tokens.append(sample['generated_tokens'])
                reference_tokens.append(sample['original_continuation_tokens'])

            if 'generated_text' in sample and 'original_continuation' in sample:
                generated_texts.append(sample['generated_text'])
                reference_texts.append(sample['original_continuation'])

            if 'top_tokens' in sample:
                top_tokens_list.append(sample['top_tokens'])

        results = {}

        # Method 1: Exact match rate
        if generated_tokens and reference_tokens:
            results["exact_match"] = self.exact_match_rate(generated_tokens, reference_tokens)

        # Method 2: ROUGE/BLEU scores
        if generated_texts and reference_texts:
            results["rouge_bleu"] = self.rouge_bleu_scores(
                generated_texts, reference_texts, generated_tokens, reference_tokens
            )

        # Method 3: Edit distance
        if generated_tokens and reference_tokens:
            results["edit_distance"] = self.edit_distance_metrics(
                generated_tokens, reference_tokens, generated_texts, reference_texts
            )

        # Method 4: Semantic similarity
        if generated_texts and reference_texts:
            results["semantic"] = self.semantic_similarity(generated_texts, reference_texts)

        # Method 5: Likelihood, PPL, loss
        if top_tokens_list and reference_tokens:
            results["likelihood"] = self.likelihood_ppl_loss_metrics(top_tokens_list, reference_tokens)

        return results


# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = MemorizationMetrics()

    # Mock sample data
    samples = [
        {
            'generated_tokens': [3544, 6303, 323, 358, 1390],
            'original_continuation_tokens': [3544, 2816, 449, 279, 1925],
            'generated_text': 'large matrix and I want',
            'original_continuation': 'large site with the main',
            'top_tokens': [
                [
                    {'token_id': 3544, 'probability': 0.8, 'rank': 1},
                    {'token_id': 2816, 'probability': 0.1, 'rank': 2}
                ],
                [
                    {'token_id': 6303, 'probability': 0.6, 'rank': 1},
                    {'token_id': 2816, 'probability': 0.3, 'rank': 2}
                ]
            ]
        }
    ]

    # Compute all metrics
    results = evaluator.compute_all_metrics_from_data(samples)

    for metric_type, metrics in results.items():
        print(f"\n{metric_type.upper()}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
