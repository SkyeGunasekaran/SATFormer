#!/usr/bin/env python3

import os
import time
import random
import logging
import argparse
import multiprocessing as mp
import numpy as np
from functools import wraps
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# Global Config & Constants

FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_SUBSET = "sample-100BT"
TOKENIZER_ID = "meta-llama/Llama-2-7b-hf"

DTYPE = np.uint16
EOS_TOKEN = 2  # Standard Llama 2 EOS

# Retry config
MAX_RETRIES = 8
RETRY_BASE_DELAY = 2.0   # seconds
RETRY_MAX_DELAY = 120.0  # seconds
RETRY_JITTER = 0.3       # fraction of delay to randomise

_worker_tokenizer = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Retry Utilities

def with_retry(max_retries=MAX_RETRIES, base_delay=RETRY_BASE_DELAY,
               max_delay=RETRY_MAX_DELAY, jitter=RETRY_JITTER,
               exceptions=(Exception,)):
    """
    Decorator: retry a function with exponential backoff + jitter.
    Surfaces the final exception if all attempts fail.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_retries:
                        log.error(
                            f"{fn.__name__} failed after {max_retries} attempts. "
                            f"Last error: {exc}"
                        )
                        raise
                    sleep_time = min(delay, max_delay)
                    sleep_time *= 1 + jitter * (2 * random.random() - 1)
                    log.warning(
                        f"{fn.__name__} attempt {attempt}/{max_retries} failed: {exc}. "
                        f"Retrying in {sleep_time:.1f}s…"
                    )
                    time.sleep(sleep_time)
                    delay = min(delay * 2, max_delay)
        return wrapper
    return decorator


# Worker Functions

@with_retry(
    max_retries=MAX_RETRIES,
    exceptions=(OSError, ConnectionError, TimeoutError, Exception),
)
def _load_tokenizer(tokenizer_id):
    return AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)


def worker_init(tokenizer_id):
    global _worker_tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        _worker_tokenizer = _load_tokenizer(tokenizer_id)
    except Exception as exc:
        # Log but don't crash — process_batch will surface the error later.
        log.error(f"Worker {os.getpid()}: tokenizer load failed: {exc}")
        _worker_tokenizer = None


def process_batch(text_batch):
    global _worker_tokenizer

    if _worker_tokenizer is None:
        # Attempt a lazy reload before giving up.
        try:
            _worker_tokenizer = _load_tokenizer(TOKENIZER_ID)
        except Exception as exc:
            log.error(f"Worker {os.getpid()}: lazy tokenizer reload failed: {exc}")
            return None

    try:
        encodings = _worker_tokenizer(
            text_batch,
            add_special_tokens=False,
            return_attention_mask=False,
        )

        all_ids = []
        for ids in encodings["input_ids"]:
            all_ids.extend(ids)
            all_ids.append(EOS_TOKEN)

        return np.array(all_ids, dtype=DTYPE)

    except Exception as exc:
        log.error(f"Worker {os.getpid()}: tokenisation error (batch dropped): {exc}")
        return None

# Dataset Loading (with retry)

@with_retry(
    max_retries=MAX_RETRIES,
    base_delay=3.0,
    exceptions=(OSError, ConnectionError, TimeoutError, Exception),
)
def load_dataset_with_retry(dataset_name, subset, seed, buffer_size=10_000):
    """Load a streaming HuggingFace dataset with exponential-backoff retry."""
    return (
        load_dataset(dataset_name, name=subset, split="train", streaming=True)
        .shuffle(seed=seed, buffer_size=buffer_size)
    )


# Resilient Batch Generator

def resilient_batch_generator(args):
    batch_size = args.batch_size
    max_stream_retries = MAX_RETRIES

    docs_seen = 0
    batch = []

    for stream_attempt in range(1, max_stream_retries + 1):
        try:
            dataset = load_dataset_with_retry(
                FINEWEB_DATASET, FINEWEB_SUBSET, args.seed
            )

            # Fast-forward past already-consumed documents.
            log.info(f"Stream attempt {stream_attempt}: skipping {docs_seen} docs…")
            stream_iter = iter(dataset)
            for _ in range(docs_seen):
                try:
                    next(stream_iter)
                except StopIteration:
                    return  # Dataset exhausted during skip — we're done.

            for item in stream_iter:
                docs_seen += 1
                batch.append(item["text"])
                if len(batch) == batch_size:
                    yield batch
                    batch = []

            # Clean exit — yield any leftover partial batch.
            if batch:
                yield batch
            return  # Done successfully.

        except (OSError, ConnectionError, TimeoutError, StopIteration) as exc:
            if stream_attempt == max_stream_retries:
                log.error(f"Stream failed permanently after {max_stream_retries} attempts.")
                raise
            delay = min(RETRY_BASE_DELAY * (2 ** stream_attempt), RETRY_MAX_DELAY)
            log.warning(
                f"Stream error on attempt {stream_attempt}: {exc}. "
                f"Reconnecting in {delay:.0f}s… "
                f"({docs_seen} docs already consumed)"
            )
            time.sleep(delay)
            # Yield the partial batch before reconnecting so progress isn't lost.
            if batch:
                yield batch
                batch = []

# Main Processing Logic

def process_fineweb(args):
    # 1. Setup Output
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")

    print(f"\n{'='*50}")
    print(f"fineweb-edu preprocessing")
    print(f"{'='*50}")
    print(f"Target:       {args.tokens / 1e9:.2f} B tokens")
    print(f"Workers:      {args.workers}")
    print(f"Output:       {train_path}, {val_path}")
    print(f"{'='*50}\n")

    # 2. Main Loop
    with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:

        with mp.Pool(
            args.workers,
            initializer=worker_init,
            initargs=(TOKENIZER_ID,),
            maxtasksperchild=500,   # recycle workers to avoid memory leaks
        ) as pool:

            total_tokens = 0
            train_tokens = 0
            val_tokens = 0
            target = int(args.tokens)
            skipped_batches = 0

            rng = np.random.default_rng(args.seed)
            pbar = tqdm(total=target, unit="tok", desc="Writing")

            try:
                iterator = pool.imap_unordered(
                    process_batch,
                    resilient_batch_generator(args),
                    chunksize=4,    # reduce round-trip overhead
                )

                for token_array in iterator:
                    # Worker returned None — batch failed, skip it.
                    if token_array is None:
                        skipped_batches += 1
                        if skipped_batches % 10 == 0:
                            log.warning(f"{skipped_batches} batches skipped so far.")
                        continue

                    if rng.random() < args.val_fraction:
                        f_val.write(token_array.tobytes())
                        val_tokens += len(token_array)
                    else:
                        f_train.write(token_array.tobytes())
                        train_tokens += len(token_array)

                    total_tokens += len(token_array)
                    pbar.update(len(token_array))

                    if total_tokens >= target:
                        log.info(f"Target reached: {total_tokens:,} tokens.")
                        break

            except KeyboardInterrupt:
                log.warning("Interrupted by user. Saving current progress…")
                pool.terminate()
            except Exception as exc:
                log.error(f"Fatal error in main loop: {exc}")
                pool.terminate()
                raise
            finally:
                pbar.close()

    # 3. Final Summary
    print(f"\nPreprocessing Complete")
    print(f"Train Tokens:    {train_tokens / 1e9:.4f} B")
    print(f"Val Tokens:      {val_tokens / 1e9:.4f} B")
    print(f"Skipped Batches: {skipped_batches}")
    print(f"Train Size:      {os.path.getsize(train_path) / (1024**3):.2f} GiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens",       type=float, default=100e6,        help="Target total tokens")
    parser.add_argument("--output",       type=str,   default="data_fineweb", help="Output directory")
    parser.add_argument("--workers",      type=int,   default=os.cpu_count(), help="CPU workers")
    parser.add_argument("--batch_size",   type=int,   default=1000,          help="Documents per batch sent to workers")
    parser.add_argument("--val_fraction", type=float, default=0.005,         help="Validation split (0.005 = 0.5%)")
    parser.add_argument("--seed",         type=int,   default=42)

    args = parser.parse_args()
    process_fineweb(args)