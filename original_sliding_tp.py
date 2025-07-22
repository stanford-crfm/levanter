from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict
import threading
import queue
import tempfile

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import torch_xla.distributed.xla_multiprocessing as xmp
from transformers import AutoModelForCausalLM, AutoTokenizer
import fsspec
import gc
import matplotlib.pyplot as plt
import wandb


def fsspec_mkdirs(dir_path, exist_ok=True):
    """
    Create a directory in a fsspec filesystem.

    Args:
        dir_path (str): The path of the directory
    """
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
logger = logging.getLogger("sliding_logits")


class Precision(Enum):
    FLOAT16 = auto()
    FLOAT32 = auto()


@dataclass
class SlidingLogitsTPConfig:
    """Configuration for tensor-parallel sliding-window forward-pass logging."""

    model_name: str
    input_path: str  # path to raw txt (local or gs://)
    output_dir: str  # directory where output shards (.npz/.npy) will be written

    # Runtime / batching --------------------------------------------------
    batch_size: int = 1  # For tensor parallel, we process one chunk at a time

    # Chunk parameters ----------------------------------------------------
    chunk_size: int = 100
    slice_length: int = 2000
    cursor_inc: int = 10

    # Tokeniser / model ---------------------------------------------------
    max_length: int = 100  # ensure model input not longer than chunk

    # Prompt / suffix split ----------------------------------------------
    prompt_tokens: int | None = None

    # Numerical precision for model weights + saved logits.
    precision: Precision = Precision.FLOAT32

    # TPU device count (set TPU_NUM_DEVICES). If None, use all visible cores.
    num_devices: int | None = None

    # If True, write uncompressed .npy files instead of .npz
    uncompress: bool = False

    # Block size for fsspec writes in bytes
    block_size: int = 64 * 1024 * 1024

    # Number of batches to accumulate before writing
    batches_per_save: int = 1

    # Background writing
    background_queue: bool = False
    num_background_writers: int = 1

    # Tensor parallel mesh shape - typically (1, num_devices) for model parallel
    mesh_shape: tuple[int, int] | None = None
    
    # Debug flag
    debug: bool = False
    
    # Plotting configuration
    create_plot: bool = True
    plot_title: str = "Sliding Logits: Maximum per-character probability"
    colormap: str = "Blues"
    figsize: tuple[int, int] = (14, 2)
    dpi: int = 300
    save_combined_arrays: bool = True
    compute_extraction_stats: bool = True
    
    # WandB configuration
    wandb_run_name: str | None = None


def _writer_loop(batch_queue: queue.Queue, cfg: SlidingLogitsTPConfig, error_list: list[Exception]) -> None:
    """Background thread function to write batches to disk."""
    thread_id = threading.get_ident()
    if cfg.debug:
        print(f"[Writer {thread_id}] Background writer thread started", flush=True)
    
    try:
        write_count = 0
        while True:
            payload = batch_queue.get()
            
            if payload is None:
                if cfg.debug:
                    print(f"[Writer {thread_id}] Received shutdown signal", flush=True)
                batch_queue.task_done()
                break
            
            write_start_time = time.time()
            data_dict, batch_path = payload
            
            if cfg.debug:
                print(f"[Writer {thread_id}] Starting write #{write_count} to {batch_path}", flush=True)
            
            try:
                with fsspec.open(batch_path, "wb", block_size=cfg.block_size) as fo:
                    if cfg.uncompress:
                        np.save(fo, data_dict, allow_pickle=True)
                    else:
                        np.savez_compressed(fo, **data_dict)
            except Exception as write_exc:
                print(f"[Writer {thread_id}] ERROR during write: {write_exc}", flush=True)
                raise write_exc
            finally:
                batch_queue.task_done()
            
            total_write_time = time.time() - write_start_time
            if cfg.debug:
                print(f"[Writer {thread_id}] Completed write #{write_count} in {total_write_time:.2f}s", flush=True)
            write_count += 1
            
    except Exception as exc:
        print(f"[Writer {thread_id}] FATAL ERROR: {exc}", flush=True)
        error_list.append(exc)
    
    if cfg.debug:
        print(f"[Writer {thread_id}] Background writer thread exiting after {write_count} writes", flush=True)


def _apply_tensor_parallel_sharding(model, mesh, debug=False):
    """Apply tensor parallel sharding to model parameters."""
    print(f"[TP] Applying tensor parallel sharding to model parameters", flush=True)
    if debug:
        print(f"[TP] Mesh shape: {mesh.shape()}", flush=True)
    
    param_count = 0
    sharded_count = 0
    
    for name, param in model.named_parameters():
        param_count += 1
        param_shape = param.shape
        if debug:
            print(f"[TP] Parameter {name}: shape={param_shape}, numel={param.numel()}", flush=True)
        
        # Apply sharding based on parameter type and shape
        if len(param_shape) >= 2:
            if param_shape[-1] >= mesh.shape()['model']:
                partition_spec = tuple(None for _ in range(len(param_shape) - 1)) + ('model',)
                xs.mark_sharding(param, mesh, partition_spec)
                sharded_count += 1
                if debug:
                    print(f"[TP] Sharded {name} with spec {partition_spec}", flush=True)
            else:
                if debug:
                    print(f"[TP] Replicated {name} (dimension too small for sharding)", flush=True)
        else:
            if debug:
                print(f"[TP] Replicated {name} (1D tensor)", flush=True)
    
    print(f"[TP] Applied sharding to {sharded_count}/{param_count} parameters", flush=True)
    return model


def create_sliding_logits_plot(cfg: SlidingLogitsTPConfig, char_max_array: np.ndarray, 
                               original_text: str, all_pz_values: list[float]) -> Dict[str, Any]:
    """
    Create character-level heatmap from char_max array.
    
    Returns:
        Dictionary with paths and summary statistics
    """
    
    print(f"[TP] Creating sliding logits plot...", flush=True)
    logger.info("Creating sliding logits plot")
    
    # Compute basic statistics
    max_prob = float(char_max_array.max())
    mean_prob = float(char_max_array.mean())
    min_prob = float(char_max_array.min())
    nonzero_count = int(np.count_nonzero(char_max_array))
    actual_length = len(char_max_array)
    
    logger.info("Character-level statistics:")
    logger.info("  Max probability: %.6f", max_prob)
    logger.info("  Mean probability: %.6f", mean_prob)
    logger.info("  Min probability: %.6f", min_prob)
    logger.info("  Non-zero positions: %d/%d (%.1f%%)", 
                nonzero_count, actual_length, 100 * nonzero_count / actual_length)
    
    # Create visualization
    try:
        fig, ax = plt.subplots(figsize=cfg.figsize)
        
        # Create single-row heatmap
        im = ax.imshow(
            char_max_array[np.newaxis, :],  # shape (1, text_len)
            cmap=cfg.colormap,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        
        ax.set_title(cfg.plot_title)
        ax.set_xlabel("Character position")
        ax.set_yticks([])  # Hide y-axis (only one row)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
        cbar.set_label("Max. suffix probability")
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(cfg.output_dir, "sliding_logits_plot.png")
        logger.info("Saving plot to %s", plot_path)
        with fsspec.open(plot_path, "wb") as f:
            plt.savefig(f, dpi=cfg.dpi, bbox_inches='tight')
        
        plt.close()
        logger.info("Plot saved successfully")
        
    except Exception as e:
        logger.error("Error creating or saving plot: %s", e)
        raise
    
    # Prepare results
    results = {
        "plot_path": plot_path,
        "text_length": actual_length,
        "max_probability": max_prob,
        "mean_probability": mean_prob,
        "min_probability": min_prob,
        "nonzero_positions": nonzero_count,
        "coverage_percent": 100 * nonzero_count / actual_length,
    }
    
    # Compute extraction statistics if requested
    if cfg.compute_extraction_stats and all_pz_values:
        logger.info("Computing extraction rate statistics...")
        
        pz_array = np.array(all_pz_values)
        
        # Compute extraction rates for different probability thresholds
        p_thresholds = [0.5, 0.9, 0.99]
        extraction_stats = {}
        
        for p in p_thresholds:
            valid_pz = pz_array[(pz_array > 0) & (pz_array < 1)]
            if len(valid_pz) > 0:
                log1m_p = np.log(1 - p)
                log1m_pz = np.log(1 - valid_pz)
                n_values = np.ceil(log1m_p / log1m_pz)
                n_values = n_values[np.isfinite(n_values)]
                
                if len(n_values) > 0:
                    extraction_stats[f"mean_n_for_p_{p}"] = float(np.mean(n_values))
                    extraction_stats[f"median_n_for_p_{p}"] = float(np.median(n_values))
                    extraction_stats[f"max_n_for_p_{p}"] = float(np.max(n_values))
        
        extraction_stats["num_pz_values"] = len(all_pz_values)
        extraction_stats["mean_pz"] = float(np.mean(pz_array))
        extraction_stats["median_pz"] = float(np.median(pz_array))
        extraction_stats["max_pz"] = float(np.max(pz_array))
        
        results["extraction_stats"] = extraction_stats
        
        logger.info("Extraction statistics computed:")
        for key, value in extraction_stats.items():
            logger.info("  %s: %s", key, value)
        
        # Save extraction stats
        if cfg.save_combined_arrays:
            stats_path = os.path.join(cfg.output_dir, "extraction_stats.npy")
            logger.info("Saving extraction statistics to %s", stats_path)
            
            with fsspec.open(stats_path, "wb") as f:
                np.save(f, extraction_stats)
            results["extraction_stats_path"] = stats_path
    
    return results


def compute_sliding_logits_tp(cfg: SlidingLogitsTPConfig) -> None:
    """Run tensor-parallel causal-LM forward pass over sliding windows and save outputs."""
    
    # Initialize wandb
    if cfg.wandb_run_name:
        wandb.init(
            name=cfg.wandb_run_name,
            config={
                "model_name": cfg.model_name,
                "input_path": cfg.input_path,
                "output_dir": cfg.output_dir,
                "batch_size": cfg.batch_size,
                "chunk_size": cfg.chunk_size,
                "slice_length": cfg.slice_length,
                "cursor_inc": cfg.cursor_inc,
                "max_length": cfg.max_length,
                "prompt_tokens": cfg.prompt_tokens,
                "precision": cfg.precision.name,
                "num_devices": cfg.num_devices,
                "mesh_shape": cfg.mesh_shape,
                "batches_per_save": cfg.batches_per_save,
                "background_queue": cfg.background_queue,
                "num_background_writers": cfg.num_background_writers,
            }
        )
        print(f"Initialized wandb run: {cfg.wandb_run_name}", flush=True)
    
    print(f" torch xla version is {torch_xla.__version__}", flush=True)
    print(f" torch xla version is {torch_xla.__version__}", flush=True)
    print(f" torch xla version is {torch_xla.__version__}", flush=True)
    print(f" XLA_USE_F16 is {os.environ.get('XLA_USE_F16')}", flush=True)
    print(f" XLA_USE_BF16 is {os.environ.get('XLA_USE_BF16')}", flush=True)
    print(f" XLA_DOWNCAST_BF16 is {os.environ.get('XLA_DOWNCAST_BF16')}", flush=True)
    print(f" XLA_SAVE_TENSORS_FILE is {os.environ.get('XLA_SAVE_TENSORS_FILE')}", flush=True)
    print(f" XLA_SAVE_TENSORS_FMT is {os.environ.get('XLA_SAVE_TENSORS_FMT')}", flush=True)

    print(f"Starting compute_sliding_logits_tp for model: {cfg.model_name}", flush=True)
    print(f"Input path: {cfg.input_path}", flush=True)
    print(f"Output directory: {cfg.output_dir}", flush=True)
    print(f"Batch size: {cfg.batch_size}, Chunk size: {cfg.chunk_size}", flush=True)
    print(f"device attributes: {xr.global_runtime_device_attributes()}", flush=True)
    logger.info(
        "Computing tensor-parallel sliding-window logits for %s using %s",
        cfg.input_path,
        cfg.model_name,
    )
    
    # Ensure output directory exists
    print("Creating output directory...", flush=True)
    fsspec_mkdirs(cfg.output_dir)
    print("Output directory created successfully", flush=True)
    
    # Configure TPU device visibility
    if cfg.num_devices is not None:
        print(f"Setting TPU_NUM_DEVICES to {cfg.num_devices}", flush=True)
        os.environ["TPU_NUM_DEVICES"] = str(cfg.num_devices)
        os.environ["PJRT_DEVICE_COUNT"] = str(cfg.num_devices)
        os.environ.pop("X_NUM_DEVICES", None)
        logger.info("Set TPU_NUM_DEVICES=%s", cfg.num_devices)
    else:
        print("Using default TPU device configuration", flush=True)
        os.environ.pop("X_NUM_DEVICES", None)
        os.environ.pop("TPU_NUM_DEVICES", None)
        os.environ.pop("PJRT_DEVICE_COUNT", None)
    
    # Since we're using tensor parallelism, we need SPMD mode
    print("Enabling SPMD mode for tensor parallelism...", flush=True)
    xr.use_spmd()
    
    # Force PJRT runtime to initialize
    print("Initializing PJRT runtime...", flush=True)
    world_size = xr.world_size()
    print(f"PJRT runtime initialized with world_size={world_size}", flush=True)
    logger.info("Process sees XR world_size=%d", world_size)
    
    # For tensor parallelism, we run everything in the main process
    # No xmp.spawn needed since we're sharding the model, not the data
    print("Running tensor-parallel worker in main process...", flush=True)
    _sliding_logits_tp_worker(0, cfg)
    print("Tensor-parallel processing completed", flush=True)


def _sliding_logits_tp_worker(index: int, cfg: SlidingLogitsTPConfig) -> None:
    """Tensor-parallel worker. Runs in main process for TP."""
    
    print(f"[Worker {index}] Starting tensor-parallel worker", flush=True)
    
    # Get device information
    num_devices = xr.global_runtime_device_count()
    print(f"[TP] Total devices available: {num_devices}", flush=True)
    
    # Create mesh for tensor parallelism
    mesh_shape = cfg.mesh_shape if cfg.mesh_shape is not None else (1, num_devices)
    print(f"[TP] Creating mesh with shape: {mesh_shape}", flush=True)
    
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ('data', 'model'))
    print(f"[TP] Created mesh with shape: {mesh.shape()}", flush=True)
    
    # Load text
    print(f"[TP] Loading text from {cfg.input_path}...", flush=True)
    fs_file = fsspec.open(cfg.input_path, "r")
    with fs_file as f:
        full_text: str = f.read()
    print(f"[TP] Loaded text with {len(full_text)} characters", flush=True)
    
    # Load tokenizer
    print(f"[TP] Loading tokenizer {cfg.model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[TP] Tokenizer loaded", flush=True)
    
    # Create chunks
    print(f"[TP] Creating sliding window chunks...", flush=True)
    chunks = chunk_text_to_sliding_window_token_chunks(
        full_text,
        tokenizer,
        chunk_size=cfg.chunk_size,
        slice_length=cfg.slice_length,
        cursor_inc=cfg.cursor_inc,
    )
    logger.info("[TP] Total generated windows: %d", len(chunks))
    print(f"[TP] Created {len(chunks)} sliding window chunks", flush=True)
    torch_xla._XLAC._xla_set_mat_mul_precision('highest')
    print(f"matmlul set to highest", flush=True)
    
    # Load model
    print(f"[TP] Loading model {cfg.model_name}...", flush=True)
    desired_dtype = torch.float16 if cfg.precision == Precision.FLOAT16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=desired_dtype)
    print(f"[TP] Model loaded with dtype {desired_dtype}", flush=True)
    
    # Move model to XLA device
    print(f"[TP] Moving model to XLA device...", flush=True)
    device = xm.xla_device()
    model.to(device)
    model.eval()
    print(f"[TP] Model moved to device {device}", flush=True)
    
    # Apply tensor parallel sharding
    model = _apply_tensor_parallel_sharding(model, mesh, debug=cfg.debug)
    print(f"[TP] Model sharding applied successfully", flush=True)
    
    # Setup output paths
    shard_path_prefix = os.path.join(cfg.output_dir, f"sliding_logits_tp")
    prompt_len = cfg.prompt_tokens if cfg.prompt_tokens is not None else cfg.chunk_size // 2
    
    # Character-level max-prob array
    text_len = len(full_text)
    char_max_local = np.zeros(text_len, dtype=np.float32)
    print(f"[TP] Initialized char_max array with length {text_len}", flush=True)
    
    # List to collect all pz values for extraction statistics
    all_pz_values = []
    
    # Setup background queue if enabled
    batch_queue: queue.Queue | None = None
    writer_threads: list[threading.Thread] = []
    writer_errors: list[Exception] = []
    
    if cfg.background_queue:
        queue_size = cfg.num_background_writers * 3
        batch_queue = queue.Queue(maxsize=queue_size)
        print(f"[TP] Setting up {cfg.num_background_writers} background writer threads", flush=True)
        
        for i in range(cfg.num_background_writers):
            t = threading.Thread(
                target=_writer_loop,
                args=(batch_queue, cfg, writer_errors),
                daemon=True,
            )
            t.start()
            writer_threads.append(t)
    
    # Process chunks
    total_chunks = len(chunks)
    start_time = time.time()
    save_counter = 0
    accum_batches = 0
    
    # Wandb logging setup
    log_interval = max(1, total_chunks // 100)  # Log every 1% of progress
    last_log_time = start_time
    chunk_times = []
    pz_values = []
    
    # Accumulation lists
    accum_logits: list[np.ndarray] = []
    accum_input_ids: list[np.ndarray] = []
    accum_start_idx: list[np.ndarray] = []
    accum_end_idx: list[np.ndarray] = []
    accum_text_len: list[np.ndarray] = []
    accum_text: list[np.ndarray] = []
    accum_pz: list[np.ndarray] = []
    
    print(f"[TP] Starting processing of {total_chunks} chunks", flush=True)
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk_start_time = time.time()
        
        if chunk_idx % 10 == 0:
            progress_percent = (chunk_idx + 1) / total_chunks * 100
            print(f"[TP] Processing chunk {chunk_idx + 1}/{total_chunks} ({progress_percent:.1f}%)", flush=True)
        
        # Tokenize
        text = chunk["text"]
        tokens = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        
        # Move to device and mark as replicated
        tokens = {k: v.to(device) for k, v in tokens.items()}
        for k, v in tokens.items():
            xs.mark_sharding(v, mesh, (None, None))  # Replicate across all devices
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**tokens)
        
        logits = outputs.logits.to(desired_dtype)
        
        # Compute P(z)
        shift_logits = logits[:, :-1, :]
        shift_labels = tokens["input_ids"][:, 1:]
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        suffix_start = max(0, prompt_len - 1)
        if suffix_start < token_lp.size(1):
            suffix_lp = token_lp[:, suffix_start:].sum(dim=-1)
            pz_value = torch.exp(suffix_lp).cpu().item()
        else:
            pz_value = 0.0
        
        # Collect pz value for extraction statistics
        all_pz_values.append(pz_value)
        pz_values.append(pz_value)
        
        # Convert to numpy
        logits_np = logits.cpu().numpy()
        
        # Create arrays for this chunk
        chunk_input_ids = np.array([chunk["input_ids"]], dtype=np.int32)
        chunk_start_idx = np.array([chunk["start_idx"]], dtype=np.int32)
        chunk_end_idx = np.array([chunk["end_idx"]], dtype=np.int32)
        chunk_text_len = np.array([chunk["text_len"]], dtype=np.int32)
        chunk_text = np.array([chunk["text"]], dtype=object)
        chunk_pz = np.array([pz_value], dtype=np.float32)
        
        # Update character-level max
        c0, c1 = chunk["start_idx"], chunk["end_idx"]
        char_max_local[c0 : c1 + 1] = np.maximum(char_max_local[c0 : c1 + 1], pz_value)
        
        # Track timing
        chunk_time = time.time() - chunk_start_time
        chunk_times.append(chunk_time)
        
        # Wandb logging
        if cfg.wandb_run_name and (chunk_idx % log_interval == 0 or chunk_idx == total_chunks - 1):
            current_time = time.time()
            elapsed_time = current_time - start_time
            progress_percent = (chunk_idx + 1) / total_chunks * 100
            
            # Compute metrics
            recent_pz = pz_values[-min(100, len(pz_values)):]  # Last 100 pz values
            recent_times = chunk_times[-min(100, len(chunk_times)):]  # Last 100 chunk times
            
            wandb.log({
                "chunk": chunk_idx + 1,
                "total_chunks": total_chunks,
                "progress_percent": progress_percent,
                "elapsed_time": elapsed_time,
                "chunks_per_second": (chunk_idx + 1) / elapsed_time if elapsed_time > 0 else 0,
                "current_pz": pz_value,
                "pz_mean": np.mean(recent_pz) if recent_pz else 0,
                "pz_max": np.max(recent_pz) if recent_pz else 0,
                "pz_min": np.min(recent_pz) if recent_pz else 0,
                "chunk_time": chunk_time,
                "avg_chunk_time": np.mean(recent_times) if recent_times else 0,
            }, step=chunk_idx + 1)
            
            last_log_time = current_time
        
        # Accumulate
        accum_logits.append(logits_np)
        accum_input_ids.append(chunk_input_ids)
        accum_start_idx.append(chunk_start_idx)
        accum_end_idx.append(chunk_end_idx)
        accum_text_len.append(chunk_text_len)
        accum_text.append(chunk_text)
        accum_pz.append(chunk_pz)
        accum_batches += 1
        
        # Save when reaching batch limit
        if accum_batches >= cfg.batches_per_save:
            out_logits = np.concatenate(accum_logits, axis=0)
            out_input_ids = np.concatenate(accum_input_ids, axis=0)
            out_start_idx = np.concatenate(accum_start_idx, axis=0)
            out_end_idx = np.concatenate(accum_end_idx, axis=0)
            out_text_len = np.concatenate(accum_text_len, axis=0)
            out_text = np.concatenate(accum_text, axis=0)
            out_pz = np.concatenate(accum_pz, axis=0)
            
            ext = "npy" if cfg.uncompress else "npz"
            batch_path = f"{shard_path_prefix}_part{save_counter}.{ext}"
            data_dict = {
                "input_ids": out_input_ids,
                "start_idx": out_start_idx,
                "end_idx": out_end_idx,
                "text_len": out_text_len,
                "text": out_text,
                "logits": out_logits,
                "pz": out_pz,
            }
            
            if cfg.background_queue:
                assert batch_queue is not None
                batch_queue.put((data_dict, batch_path))
            else:
                with fsspec.open(batch_path, "wb", block_size=cfg.block_size) as fo:
                    if cfg.uncompress:
                        np.save(fo, data_dict, allow_pickle=True)
                    else:
                        np.savez_compressed(fo, **data_dict)
            
            save_counter += 1
            accum_batches = 0
            accum_logits.clear()
            accum_input_ids.clear()
            accum_start_idx.clear()
            accum_end_idx.clear()
            accum_text_len.clear()
            accum_text.clear()
            accum_pz.clear()
        
        # Cleanup
        del logits, tokens, outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        xm.mark_step()
        gc.collect()
    
    # Save any remaining data
    if accum_batches > 0:
        out_logits = np.concatenate(accum_logits, axis=0)
        out_input_ids = np.concatenate(accum_input_ids, axis=0)
        out_start_idx = np.concatenate(accum_start_idx, axis=0)
        out_end_idx = np.concatenate(accum_end_idx, axis=0)
        out_text_len = np.concatenate(accum_text_len, axis=0)
        out_text = np.concatenate(accum_text, axis=0)
        out_pz = np.concatenate(accum_pz, axis=0)
        
        ext = "npy" if cfg.uncompress else "npz"
        batch_path = f"{shard_path_prefix}_part{save_counter}.{ext}"
        data_dict = {
            "input_ids": out_input_ids,
            "start_idx": out_start_idx,
            "end_idx": out_end_idx,
            "text_len": out_text_len,
            "text": out_text,
            "logits": out_logits,
            "pz": out_pz,
        }
        
        if cfg.background_queue:
            assert batch_queue is not None
            batch_queue.put((data_dict, batch_path))
        else:
            with fsspec.open(batch_path, "wb", block_size=cfg.block_size) as fo:
                if cfg.uncompress:
                    np.save(fo, data_dict, allow_pickle=True)
                else:
                    np.savez_compressed(fo, **data_dict)
    
    # Shutdown background writers
    if cfg.background_queue and batch_queue is not None:
        print(f"[TP] Shutting down background writers...", flush=True)
        
        for i in range(len(writer_threads)):
            batch_queue.put(None)
        
        batch_queue.join()
        
        for t in writer_threads:
            t.join()
        
        if writer_errors:
            print(f"[TP] ERROR: {len(writer_errors)} writer errors occurred", flush=True)
            raise writer_errors[0]
    
    # Final timing
    total_time = time.time() - start_time
    print(f"[TP] Completed {total_chunks} chunks in {total_time:.1f}s", flush=True)
    
    # Final wandb logging
    if cfg.wandb_run_name:
        wandb.log({
            "total_time": total_time,
            "total_chunks": total_chunks,
            "chunks_per_second": total_chunks / total_time if total_time > 0 else 0,
        }, step=total_chunks)
    # Write character max array
    cm_path = os.path.join(cfg.output_dir, f"char_max_tp.npy")
    print(f"[TP] Writing char_max to {cm_path}...", flush=True)
    with fsspec.open(cm_path, "wb") as fo:
        np.save(fo, char_max_local)
   
    print("[TP] Wrote char_max to %s", cm_path)
   
   # Save combined char_max array if requested
    if cfg.save_combined_arrays:
        combined_path = os.path.join(cfg.output_dir, "char_max_combined.npy")
        print(f"[TP] Writing combined char_max to {combined_path}...", flush=True)
        with fsspec.open(combined_path, "wb") as fo:
            np.save(fo, char_max_local)
        print("[TP] Wrote combined char_max to %s", combined_path)
   
       # Create plot if requested
    if cfg.create_plot:
        print(f"[TP] Creating visualization plot...", flush=True)
        plot_results = create_sliding_logits_plot(cfg, char_max_local, full_text, all_pz_values)
        
        # Log plot results to wandb
        if cfg.wandb_run_name:
            wandb.log({
                "plot_path": plot_results.get("plot_path", ""),
                "text_length": plot_results.get("text_length", 0),
                "max_probability": plot_results.get("max_probability", 0),
                "mean_probability": plot_results.get("mean_probability", 0),
                "min_probability": plot_results.get("min_probability", 0),
                "nonzero_positions": plot_results.get("nonzero_positions", 0),
                "coverage_percent": plot_results.get("coverage_percent", 0),
            }, step=total_chunks + 1)
            
            # Log extraction stats if available
            if "extraction_stats" in plot_results:
                for key, value in plot_results["extraction_stats"].items():
                    wandb.log({f"extraction_{key}": value}, step=total_chunks + 1)
        
        print(f"[TP] Plot results:", flush=True)
        for key, value in plot_results.items():
            if isinstance(value, dict):
                print(f"  {key}:", flush=True)
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}", flush=True)
            else:
                print(f"  {key}: {value}", flush=True)
    
    if cfg.wandb_run_name and os.path.exists("/tmp/xla_dumps"):
        print(f"[TP] Saving XLA dumps to wandb", flush=True)
        wandb.save("/tmp/xla_dumps/*", base_path="/tmp")
    else:
        print(f" [WARNING] [TP] No XLA dumps found! \n", flush=True)
        print(f" [WARNING] [TP] No XLA dumps found! \n", flush=True)
        print(f" [WARNING] [TP] No XLA dumps found! \n", flush=True)

    # Final wandb cleanup
    if cfg.wandb_run_name:
        wandb.finish()
        print(f"[TP] Wandb run completed and logged", flush=True)
    
    print(f"[TP] Worker completed successfully", flush=True)


if __name__ == "__main__":
    cfg = SlidingLogitsTPConfig(
        model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-1-70B",
        input_path="gs://marin-us-central2/documents/books_txt/gatsby.txt",
        output_dir="gs://marin-us-east1/debug/sliding_window_tp",
        batch_size=1,
        chunk_size=100,
        slice_length=2000,
        cursor_inc=10,
        max_length=100,
        prompt_tokens=50,
        precision=Precision.FLOAT32,
        num_devices=None,
        uncompress=False,
        block_size=64 * 1024 * 1024,
        batches_per_save=10,
        background_queue=True,
        num_background_writers=2,
        mesh_shape=(1, 16),
        debug=False,
        # Plotting configuration
        create_plot=True,
        plot_title="Tensor-Parallel Sliding Logits: Great Gatsby Character Analysis",
        colormap="Blues",
        figsize=(20, 3),
        dpi=300,
        save_combined_arrays=True,
        compute_extraction_stats=True,
        # WandB configuration
        wandb_run_name="llama3-70b-gatsby-tp-sliding-fp32-east1_dont_die",  # Set your desired run name here
    )

    compute_sliding_logits_tp(cfg)