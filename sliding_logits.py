from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict
from enum import Enum, auto

import ray
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import fsspec
import datasets
import pyarrow as pa
import pyarrow.parquet as pq
import time
from concurrent.futures import ProcessPoolExecutor


def fsspec_mkdirs(dir_path, exist_ok=True):
    """
    Create a directory in a fsspec filesystem.

    Args:
        dir_path (str): The path of the directory
    """

    # Use fsspec to create the directory
    fs = fsspec.core.url_to_fs(dir_path)[0]
    fs.makedirs(dir_path, exist_ok=exist_ok)

def chunk_text_to_sliding_window_token_chunks(
    text: str,
    tokenizer,
    *,
    chunk_size: int = 100,
    slice_length: int = 2000,
    cursor_inc: int = 10,
) -> list[Dict[str, Any]]:
    """Tokenise *text* into overlapping `chunk_size`-token windows.

    Replicates the logic in ``careless.py`` almost verbatim but drops the
    torch-specific bits.  Returns a list of dictionaries with keys:

    ``input_ids``          – *list[int]* of length ``chunk_size``
    ``start_idx``          – start character index in *text*
    ``end_idx``            – end character index (inclusive) in *text*
    ``text``               – decoded chunk text (useful for debugging)
    ``attention_mask``     – list[int] same length as ``input_ids``
    ``text_len``           – length of decoded text in characters
    """

    all_chunks: list[Dict[str, Any]] = []
    text_cursor = 0
    text_len = len(text)

    progress_markers = {i for i in range(10, 101, 10)}

    while text_cursor < text_len:
        start_idx = text_cursor
        end_idx_plus_one = min(text_cursor + slice_length, text_len)
        text_slice = text[start_idx:end_idx_plus_one]

        enc = tokenizer(text_slice, add_special_tokens=False, return_attention_mask=True)
        input_ids: list[int] = enc["input_ids"][:chunk_size]
        attention_mask: list[int] = enc.get("attention_mask", [1] * len(input_ids))[:chunk_size]

        if len(input_ids) == chunk_size:
            decoded_chunk = tokenizer.decode(input_ids, skip_special_tokens=True)
            decoded_len = len(decoded_chunk)
            all_chunks.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "start_idx": start_idx,
                    "end_idx": start_idx + decoded_len - 1,
                    "text_len": decoded_len,
                    "text": decoded_chunk,
                }
            )

        text_cursor += cursor_inc
        pct_complete = int(100 * text_cursor / text_len)
        if pct_complete in progress_markers:
            logging.getLogger(__name__).info("Sliding-window progress: %s%%", pct_complete)
            progress_markers.remove(pct_complete)

    return all_chunks


# ---------------------------------------------------------------------------
# Logging & Enums
# ---------------------------------------------------------------------------
logger = logging.getLogger("ray")


class Precision(Enum):
    FLOAT16 = auto()
    FLOAT32 = auto()


@dataclass
class SlidingLogitsConfig:
    """Configuration for sliding-window forward-pass logging."""

    model_name: str
    input_path: str  # path to raw txt (local or gs://)
    output_dir: str  # directory where parquet + plot will be written

    # Runtime / batching --------------------------------------------------
    batch_size: int = 8
    memory_gb: int = 10

    # Chunk parameters ----------------------------------------------------
    chunk_size: int = 100
    slice_length: int = 2000
    cursor_inc: int = 10

    # Tokeniser / model ---------------------------------------------------
    max_length: int = 100  # ensure model input not longer than chunk

    # Prompt / suffix split ----------------------------------------------
    # Number of tokens treated as the prompt; if None, defaults to
    # `chunk_size // 2` (50 / 50 split).
    prompt_tokens: int | None = None

    # Numerical precision for model weights + saved logits.
    precision: Precision = Precision.FLOAT32

    # TPU device count (set TPU_NUM_DEVICES). If None, use all visible cores.
    num_devices: int | None = None

    # CPU offload for building and writing
    cpu_workers: int = 2
    row_group_size: int = 128
    compression: str = "zstd"


# Decorator to ensure TPU lockfile cleanup in case of errors
def compute_sliding_logits(cfg: SlidingLogitsConfig) -> None:
    """Run causal-LM forward pass over sliding windows and save outputs."""

    print(f"Starting compute_sliding_logits for model: {cfg.model_name}", flush=True)
    print(f"Input path: {cfg.input_path}", flush=True)
    print(f"Output directory: {cfg.output_dir}", flush=True)
    print(f"Batch size: {cfg.batch_size}, Chunk size: {cfg.chunk_size}", flush=True)

    logger.info(
        "Computing sliding-window logits for %s using %s",
        cfg.input_path,
        cfg.model_name,
    )   

    # Ensure output directory exists (works for GCS/local)
    print("Creating output directory...", flush=True)
    fsspec_mkdirs(cfg.output_dir)
    print("Output directory created successfully", flush=True)

    # ------------------------------------------------------------------
    # Configure TPU device visibility *before* importing torch_xla.
    # ------------------------------------------------------------------
    if cfg.num_devices is not None:
        print(f"Setting TPU_NUM_DEVICES to {cfg.num_devices}", flush=True)
        os.environ["TPU_NUM_DEVICES"] = str(cfg.num_devices)
        os.environ["PJRT_DEVICE_COUNT"] = str(cfg.num_devices)
        os.environ.pop("X_NUM_DEVICES", None)
        logger.info("Set TPU_NUM_DEVICES=%s", cfg.num_devices)
    else:
        # Default: expose all chips on the host.  Overwrite any preset value
        # (cluster base image often sets TPU_NUM_DEVICES=1).
        print("Using default TPU device configuration", flush=True)
        os.environ.pop("X_NUM_DEVICES", None)
        if "TPU_NUM_DEVICES" in os.environ:
            logger.info("Clearing pre-existing TPU_NUM_DEVICES=%s", os.environ["TPU_NUM_DEVICES"])
        os.environ.pop("TPU_NUM_DEVICES", None)
        os.environ.pop("PJRT_DEVICE_COUNT", None)

    # Lazy import AFTER env vars are settled -----------------------------------
    print("Importing torch_xla modules...", flush=True)
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.runtime as xr
    print("torch_xla modules imported successfully", flush=True)

    # Force PJRT runtime to initialise now so that xr.world_size() reflects
    # the topology.  This *must* happen after we set TPU_NUM_DEVICES.
    print("Initializing PJRT runtime...", flush=True)
    world_size = xr.world_size()  # triggers runtime init if not yet initialised
    print(f"PJRT runtime initialized with world_size={world_size}", flush=True)
    logger.info("Parent process sees XR world_size=%d", world_size)

    # ------------------------------------------------------------------
    # All heavy lifting happens in _sliding_logits_worker defined at module
    # scope (so it is picklable by multiprocessing).
    # ------------------------------------------------------------------
    print(f"Spawning {world_size} worker processes...", flush=True)
    xmp.spawn(_sliding_logits_worker, args=(cfg,), nprocs=world_size, start_method="fork")
    print("All worker processes completed", flush=True)


# ---------------------------------------------------------------------------
# Low-level worker -----------------------------------------------------------
# ---------------------------------------------------------------------------


def _sliding_logits_worker(index: int, cfg: "SlidingLogitsConfig") -> None:  # type: ignore
    """Per-XLA-core worker. Runs inside torch-xla xmp.spawn process."""

    print(f"[Core {index}] Worker started", flush=True)

    # Import torch_xla *inside* worker process, after PJRT runtime decided on
    # device topology.
    print(f"[Core {index}] Importing torch_xla modules...", flush=True)
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch
    print(f"[Core {index}] torch_xla modules imported", flush=True)

    # ------------------------------------------------------------------
    # 1. Load raw text --------------------------------------------------
    # ------------------------------------------------------------------
    print(f"[Core {index}] Loading text from {cfg.input_path}...", flush=True)
    fs_file = fsspec.open(cfg.input_path, "r")
    with fs_file as f:
        full_text: str = f.read()
    print(f"[Core {index}] Loaded text with {len(full_text)} characters", flush=True)

    print(f"[Core {index}] Loading tokenizer {cfg.model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[Core {index}] Tokenizer loaded", flush=True)

    print(f"[Core {index}] Creating sliding window chunks...", flush=True)
    chunks = chunk_text_to_sliding_window_token_chunks(
        full_text,
        tokenizer,
        chunk_size=cfg.chunk_size,
        slice_length=cfg.slice_length,
        cursor_inc=cfg.cursor_inc,
    )
    logger.info("[Core %d] Total generated windows: %d", index, len(chunks))
    print(f"[Core {index}] Created {len(chunks)} sliding window chunks", flush=True)

    print(f"[Core {index}] Loading model {cfg.model_name}...", flush=True)
    desired_dtype = torch.float16 if cfg.precision == Precision.FLOAT16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=desired_dtype)
    print(f"[Core {index}] Model loaded with dtype {desired_dtype}", flush=True)

    # Shard across world size so each XLA core gets a slice.
    world_size = xr.world_size()
    chunks_shard = chunks[index :: world_size]
    logger.info("[Core %d] Shard size: %d windows", index, len(chunks_shard))
    print(f"[Core {index}] Processing {len(chunks_shard)} chunks (shard from {len(chunks)} total)", flush=True)

    print(f"[Core {index}] Moving model to XLA device...", flush=True)
    device = xm.xla_device()
    model.to(device)
    model.eval()
    print(f"[Core {index}] Model moved to device {device}", flush=True)

    shard_path = os.path.join(cfg.output_dir, f"sliding_logits_{index}.parquet")
    print(f"[Core {index}] Will write results to {shard_path}", flush=True)

    # Build PyArrow schema (logits as list<list<float16|float32>>)
    value_type = pa.float16() if cfg.precision == Precision.FLOAT16 else pa.float32()
    schema = pa.schema(
        [
            ("input_ids", pa.list_(pa.int32())),
            ("start_idx", pa.int32()),
            ("end_idx", pa.int32()),
            ("text_len", pa.int32()),
            ("text", pa.string()),
            ("logits", pa.list_(pa.list_(value_type))),
            ("pz", pa.float32()),
        ]
    )

    # Open ParquetWriter on remote path via Arrow filesystem abstraction
    print(f"[Core {index}] Opening ParquetWriter...", flush=True)
    filesystem, path_within_fs = pa.fs.FileSystem.from_uri(shard_path)
    writer = pq.ParquetWriter(
        path_within_fs,
        schema,
        filesystem=filesystem,
        compression=cfg.compression
    )
    print(f"[Core {index}] ParquetWriter opened with compression={cfg.compression}", flush=True)
    # Spawn a pool to build and write batches without blocking the TPU loop
    executor = ProcessPoolExecutor(max_workers=cfg.cpu_workers)
    futures: list = []

    prompt_len = cfg.prompt_tokens if cfg.prompt_tokens is not None else cfg.chunk_size // 2

    # Per-core character-level max-prob array
    text_len = len(full_text)
    char_max_local = np.zeros(text_len, dtype=np.float32)
    print(f"[Core {index}] Initialized char_max_local array with {text_len} elements", flush=True)

    # ------------------------------------------------------------------
    # Helper timing decorator and per-batch processing functions
    # ------------------------------------------------------------------

    def _timed(fn):
        """Decorator to measure execution time of helper functions."""

        def wrapper(*args, **kwargs):
            _start = time.time()
            result = fn(*args, **kwargs)
            duration = time.time() - _start
            print(f"[Core {index}] {fn.__name__} took {duration:.3f}s", flush=True)
            return result

        return wrapper

    @_timed
    def _compute_pz_batch(local_logits, local_tokens):
        """Compute P(z) for each example in the batch."""

        shift_logits = local_logits[:, :-1, :]
        shift_labels = local_tokens["input_ids"][:, 1:].cpu()
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        suffix_start = max(0, prompt_len - 1)
        if suffix_start < token_lp.size(1):
            suffix_lp = token_lp[:, suffix_start:].sum(dim=-1)
            return torch.exp(suffix_lp).tolist()
        else:
            return [0.0] * local_logits.size(0)

    @_timed
    def _build_table(batch_chunks, logits_np, pz_batch, is_fp16):
        import pyarrow as pa
        # Build column vectors in one go
        ids_col = pa.array([ch["input_ids"] for ch in batch_chunks], type=pa.list_(pa.int32()))
        start_col = pa.array([ch["start_idx"] for ch in batch_chunks], type=pa.int32())
        end_col = pa.array([ch["end_idx"] for ch in batch_chunks], type=pa.int32())
        textlen_col = pa.array([ch["text_len"] for ch in batch_chunks], type=pa.int32())
        text_col = pa.array([ch["text"] for ch in batch_chunks], type=pa.string())
        pz_col = pa.array(pz_batch, type=pa.float32())
        # Determine inner float type
        value_type = pa.float16() if is_fp16 else pa.float32()
        logits_type = pa.list_(pa.list_(value_type))
        # Convert numpy logits to nested list
        logits_list = logits_np.tolist()
        logits_col = pa.array(logits_list, type=logits_type)
        return pa.Table.from_arrays(
            [ids_col, start_col, end_col, textlen_col, text_col, logits_col, pz_col],
            names=["input_ids","start_idx","end_idx","text_len","text","logits","pz"]
        )

    print(f"[Core {index}] Starting to process {len(chunks_shard)} chunks", flush=True)
    for batch_start in range(0, len(chunks_shard), cfg.batch_size):
        print(f"[Core {index}] Processing batch {batch_start//cfg.batch_size + 1}/{(len(chunks_shard) + cfg.batch_size - 1)//cfg.batch_size} (chunks {batch_start}-{min(batch_start + cfg.batch_size, len(chunks_shard))})", flush=True)
        batch_chunks = chunks_shard[batch_start : batch_start + cfg.batch_size]
        texts = [c["text"] for c in batch_chunks]

        print(f"[Core {index}] Tokenizing batch...", flush=True)
        tokens = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        print(f"[Core {index}] Running forward pass...", flush=True)
        with torch.no_grad():
            outputs = model(**tokens)

        print(f"[Core {index}] Processing outputs...", flush=True)
        logits = outputs.logits.to(desired_dtype).cpu()

        # ------------------------------------------------------------------
        # Helper-function pipeline
        # ------------------------------------------------------------------

        pz_batch = _compute_pz_batch(logits, tokens)
        logits_np = logits.cpu().numpy()
        # offload build + write to CPU pool
        future = executor.submit(
            _build_table,
            batch_chunks,
            logits_np,
            pz_batch,
            cfg.precision == Precision.FLOAT16
        )
        futures.append(future)
        # free TPU tensors and advance
        del logits, tokens, outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        xm.mark_step()
        print(f"[Core {index}] Batch {batch_start//cfg.batch_size + 1} submitted to CPU pool", flush=True)

    # wait for all CPU tasks to finish
    executor.shutdown(wait=True)
    print(f"[Core {index}] All CPU tasks completed", flush=True)
    # close writer
    writer.close()
    print(f"[Core {index}] ParquetWriter closed", flush=True)

    # Write per-core char_max array directly to GCS
    cm_part_path = os.path.join(cfg.output_dir, f"char_max_part_{index}.npy")
    print(f"[Core {index}] Writing char_max part to {cm_part_path}...", flush=True)
    with fsspec.open(cm_part_path, "wb") as fo:
        np.save(fo, char_max_local)
    logger.info("[Core %d] Wrote char_max part to %s", index, cm_part_path)
    print(f"[Core {index}] Worker completed successfully", flush=True)


if __name__ == "__main__":
    cfg = SlidingLogitsConfig(
        model_name="meta-llama/Meta-Llama-3.1-8B",
        input_path="gs://marin-us-central2/documents/books_txt/gatsby.txt",
        output_dir="gs://marin-us-central2/debug/sliding_window",  # executor will create a hashed directory
        batch_size=16,
        memory_gb=16,
        chunk_size=100,
        slice_length=2000,
        cursor_inc=10,
        max_length=100,
        prompt_tokens=50,
        precision=Precision.FLOAT16,
        num_devices=4,
        cpu_workers=4,
        row_group_size=128,
        compression="zstd",
    )

    compute_sliding_logits(cfg)