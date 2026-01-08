"""
Ultra-fast dataset download script - directly take the first N samples with no sampling overhead
"""
import os
import json
import pickle
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import argparse
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import hashlib

class FastDataDownloader:
    def __init__(self,
                 output_dir: str = "data/raw",
                 cache_dir: str = ".download_cache",
                 max_workers: int = 4,
                 use_mirror: bool = True):
        """
        Initialize the fast downloader

        Parameters:
            output_dir: Output directory
            cache_dir: Cache directory (for resume/checkpointing)
            max_workers: Maximum number of parallel download threads
            use_mirror: Whether to use a domestic mirror site
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.max_workers = max_workers

        # Configure mirror endpoint
        if use_mirror:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            print("使用 HuggingFace 镜像站: https://hf-mirror.com")

        # Create required directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, dataset_name: str, subset_name: str) -> Path:
        """Get checkpoint file path"""
        checkpoint_id = hashlib.md5(f"{dataset_name}_{subset_name}".encode()).hexdigest()[:8]
        return self.cache_dir / f"checkpoint_{checkpoint_id}.pkl"

    def _load_checkpoint(self, checkpoint_path: Path) -> Optional[int]:
        """Load checkpoint and return the downloaded count"""
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                downloaded_count = checkpoint.get('downloaded_count', 0)
                print(f"从检查点恢复: 已下载 {downloaded_count} 条")
                return downloaded_count
            except Exception as e:
                print(f"加载检查点失败: {e}")
                return 0
        return 0

    def _save_checkpoint(self, checkpoint_path: Path, downloaded_count: int):
        """Save checkpoint"""
        try:
            checkpoint_data = {'downloaded_count': downloaded_count}
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        except Exception as e:
            print(f"保存检查点失败: {e}")

    def download_top_n(
        self,
        dataset_name: str,
        subset_name: str,
        target_count: int = 5000000,
        batch_size: int = 5000  # Increase batch size to reduce I/O
    ) -> str:
        """
        Directly download the first N samples (fastest approach)
        """
        output_path = self.output_dir / f"{subset_name}_top{target_count//1000000}M.jsonl"
        checkpoint_path = self._get_checkpoint_path(dataset_name, subset_name)

        # Check if already complete
        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_count = sum(1 for _ in f)
            if existing_count >= target_count:
                print(f"{subset_name} 已存在完整数据 ({existing_count} 条)，跳过下载")
                return str(output_path)

        # Load checkpoint
        downloaded_count = self._load_checkpoint(checkpoint_path)

        print(f"正在下载 {dataset_name} 的 {subset_name} 子集前 {target_count} 条数据...")

        try:
            # Stream-load dataset
            dataset = load_dataset(
                dataset_name,
                subset_name,
                split="train",
                streaming=True
            )

            # If resuming, skip already downloaded samples
            if downloaded_count > 0:
                print(f"跳过前 {downloaded_count} 条已下载的数据...")
                dataset = dataset.skip(downloaded_count)
                remaining_count = target_count - downloaded_count
            else:
                remaining_count = target_count

            # Open file for append write
            mode = 'a' if downloaded_count > 0 else 'w'
            with open(output_path, mode, encoding="utf-8") as f:
                batch_buffer = []

                with tqdm(
                    total=remaining_count,
                    initial=0,
                    desc=f"下载 {subset_name}",
                    unit="条"
                ) as pbar:

                    for idx, item in enumerate(dataset):
                        if idx >= remaining_count:
                            break

                        # Simplify data structure to reduce processing overhead
                        data_item = {
                            "text": item.get("text", ""),
                            "source": subset_name,
                            "idx": downloaded_count + idx
                        }

                        batch_buffer.append(data_item)

                        # Batch write to reduce I/O overhead
                        if len(batch_buffer) >= batch_size:
                            for data in batch_buffer:
                                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                            f.flush()  # Ensure data is flushed to disk

                            # Update progress
                            current_downloaded = downloaded_count + idx + 1
                            pbar.update(len(batch_buffer))
                            pbar.set_postfix({
                                'downloaded': f"{current_downloaded:,}",
                                'speed': f"{len(batch_buffer)/(time.time() - pbar.last_print_t if pbar.last_print_t else 1):.0f}/s"
                            })

                            # Save checkpoint
                            self._save_checkpoint(checkpoint_path, current_downloaded)
                            batch_buffer = []

                    # Write remaining data
                    if batch_buffer:
                        for data in batch_buffer:
                            f.write(json.dumps(data, ensure_ascii=False) + "\n")
                        pbar.update(len(batch_buffer))
                        final_count = downloaded_count + len(batch_buffer) + (idx + 1 - len(batch_buffer))
                        self._save_checkpoint(checkpoint_path, final_count)

        except KeyboardInterrupt:
            print(f"\n下载被中断，当前进度已保存到检查点")
            return None
        except Exception as e:
            print(f"下载出错: {e}")
            return None

        # Validate download results
        with open(output_path, 'r', encoding='utf-8') as f:
            actual_count = sum(1 for _ in f)

        print(f"下载完成: {actual_count:,} 条数据保存到 {output_path}")

        # Cleanup checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print("清理检查点文件")

        return str(output_path)

    def download_with_limit(
        self,
        dataset_name: str,
        subset_name: str,
        target_count: int = 5000000,
        skip_count: int = 0,  # Can skip the first N samples
        batch_size: int = 5000
    ) -> str:
        """
        Download a specific range: (skip_count, skip_count + target_count)
        """
        output_path = self.output_dir / f"{subset_name}_skip{skip_count//1000000}M_take{target_count//1000000}M.jsonl"

        print(f"正在下载 {dataset_name} 的 {subset_name} 子集")
        print(f"跳过前 {skip_count:,} 条，下载接下来的 {target_count:,} 条")

        try:
            dataset = load_dataset(
                dataset_name,
                subset_name,
                split="train",
                streaming=True
            )

            # Skip a specified number of samples
            if skip_count > 0:
                print(f"跳过前 {skip_count:,} 条数据...")
                dataset = dataset.skip(skip_count)

            # Download data
            with open(output_path, 'w', encoding="utf-8") as f:
                batch_buffer = []

                with tqdm(total=target_count, desc=f"下载 {subset_name}", unit="条") as pbar:
                    for idx, item in enumerate(dataset):
                        if idx >= target_count:
                            break

                        data_item = {
                            "text": item.get("text", ""),
                            "source": subset_name,
                            "idx": skip_count + idx
                        }

                        batch_buffer.append(data_item)

                        if len(batch_buffer) >= batch_size:
                            for data in batch_buffer:
                                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                            f.flush()

                            pbar.update(len(batch_buffer))
                            batch_buffer = []

                    # Write remaining data
                    if batch_buffer:
                        for data in batch_buffer:
                            f.write(json.dumps(data, ensure_ascii=False) + "\n")
                        pbar.update(len(batch_buffer))

        except Exception as e:
            print(f"下载出错: {e}")
            return None

        print(f"下载完成: {output_path}")
        return str(output_path)

    def parallel_download(
        self,
        datasets_to_download: List[tuple],
        target_count: int = 5000000
    ) -> List[str]:
        """
        Download multiple datasets in parallel
        """
        output_paths = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_dataset = {
                executor.submit(
                    self.download_top_n,
                    dataset_name,
                    subset_name,
                    target_count
                ): (dataset_name, subset_name)
                for dataset_name, subset_name in datasets_to_download
            }

            for future in as_completed(future_to_dataset):
                dataset_name, subset_name = future_to_dataset[future]
                try:
                    output_path = future.result()
                    if output_path:
                        output_paths.append(output_path)
                        print(f"✅ 成功下载: {dataset_name}/{subset_name}")
                except Exception as e:
                    print(f"❌ 下载失败 {dataset_name}/{subset_name}: {e}")

        return output_paths

    def estimate_download_time(
        self,
        dataset_name: str,
        subset_name: str,
        sample_size: int = 1000
    ):
        """
        Estimate download time
        """
        print(f"正在测试 {dataset_name}/{subset_name} 的下载速度...")

        start_time = time.time()
        dataset = load_dataset(dataset_name, subset_name, split="train", streaming=True)

        count = 0
        for item in dataset:
            count += 1
            if count >= sample_size:
                break

        elapsed = time.time() - start_time
        speed = count / elapsed

        print(f"测试下载速度: {speed:.0f} 条/秒")

        # Estimate total time
        for target in [1000000, 5000000, 10000000]:
            estimated_time = target / speed
            print(f"下载 {target:,} 条预计需要: {estimated_time/60:.1f} 分钟")


def main():
    parser = argparse.ArgumentParser(description="超快数据下载脚本")
    parser.add_argument("--target_count", type=int, default=5000000, help="下载数据条数")
    parser.add_argument("--output_dir", type=str, default="../../data/pretraining_test_data", help="输出目录")
    parser.add_argument("--cache_dir", type=str, default="../../data/download_cache", help="缓存目录")
    parser.add_argument("--max_workers", type=int, default=3, help="最大并行下载数")
    parser.add_argument("--batch_size", type=int, default=5000, help="批处理大小")
    parser.add_argument("--sequential", action="store_true", help="顺序下载")
    parser.add_argument("--no_mirror", action="store_true", help="不使用镜像站")
    parser.add_argument("--estimate", action="store_true", help="估算下载时间")
    parser.add_argument("--skip_count", type=int, default=0, help="跳过前N条数据")

    args = parser.parse_args()

    # Create downloader
    downloader = FastDataDownloader(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        max_workers=args.max_workers,
        use_mirror=not args.no_mirror
    )

    # Define datasets to download
    datasets_to_download = [
        # ("allenai/dolmino-mix-1124", "stackexchange"),
        ("allenai/olmo-mix-1124", "wiki"),
        ("allenai/olmo-mix-1124", "dclm")
    ]

    # If only estimating time
    if args.estimate:
        for dataset_name, subset_name in datasets_to_download:
            downloader.estimate_download_time(dataset_name, subset_name)
        return

    print(f"开始下载 {len(datasets_to_download)} 个数据集")
    print(f"每个数据集下载: {args.target_count:,} 条 (前{args.target_count//1000000}M条)")
    print(f"输出目录: {args.output_dir}")
    print(f"并行数: {args.max_workers if not args.sequential else 1}")
    print(f"批处理大小: {args.batch_size}")
    print("=" * 60)

    start_time = time.time()

    if args.sequential:
        # Sequential download
        for dataset_name, subset_name in datasets_to_download:
            if args.skip_count > 0:
                downloader.download_with_limit(
                    dataset_name, subset_name, args.target_count, args.skip_count
                )
            else:
                downloader.download_top_n(
                    dataset_name, subset_name, args.target_count, args.batch_size
                )
    else:
        # Parallel download
        downloader.parallel_download(datasets_to_download, args.target_count)

    elapsed_time = time.time() - start_time
    total_data = args.target_count * len(datasets_to_download)

    print(f"\n🎉 下载完成!")
    print(f"总耗时: {elapsed_time:.1f} 秒 ({elapsed_time/60:.1f} 分钟)")
    print(f"总数据量: {total_data:,} 条")
    print(f"平均速度: {total_data/elapsed_time:.0f} 条/秒")
    print(f"平均每个数据集: {elapsed_time/len(datasets_to_download):.1f} 秒")


if __name__ == "__main__":
    main()
