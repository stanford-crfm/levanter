from __future__ import annotations

import dataclasses
import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional

import fsspec
import levanter
from levanter.main.eval_careless_lm import EvalCarelessLmConfig, main as eval_book


@dataclass
class BookConfig:
    """Overrides for a single book evaluation."""

    txt_path: str
    plot_path: Optional[str] = None
    histogram_path: Optional[str] = None
    pz_data_path: Optional[str] = None
    book_title: str = "Book"
    chunk_size: Optional[int] = None
    slice_length: Optional[int] = None
    prompt_tokens: Optional[int] = None
    cursor_inc_chars: Optional[int] = None
    token_mode: Optional[bool] = None
    cursor_inc_tokens: Optional[int] = None
    eval_batch_size: Optional[int] = None


@dataclass
class MultiBookEvalConfig:
    """Configuration for running ``eval_careless_lm`` over many books."""

    base_eval: EvalCarelessLmConfig = field(default_factory=EvalCarelessLmConfig)
    books: Dict[str, BookConfig] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(cfg: MultiBookEvalConfig):
    """Run careless suffix evaluation for each book listed in ``cfg``."""

    for name, book in cfg.books.items():
        # Copy the base configuration and apply overrides for this book
        book_cfg = dataclasses.replace(cfg.base_eval)
        for field_name, value in dataclasses.asdict(book).items():
            if value is not None:
                setattr(book_cfg, field_name, value)

        # Ensure a distinct plot title if none provided explicitly
        if book_cfg.book_title == "Book":
            book_cfg.book_title = name

        # Derive default output filenames when not provided
        if book.plot_path is None:
            book_cfg.plot_path = f"bar_plot_max_pz_{book_cfg.book_title}.png"
        if book.histogram_path is None:
            book_cfg.histogram_path = f"pz_distribution_histogram_{book_cfg.book_title}.png"
        if book.pz_data_path is None:
            book_cfg.pz_data_path = f"pz_data_{book_cfg.book_title}.npz"

        # Construct per-book output directory using base path and timestamp
        ts = datetime.datetime.now().strftime("%Y%m%d%H%M")
        base = cfg.base_eval.output_base_path.rstrip("/")
        book_cfg.output_base_path = f"{base}/{book_cfg.book_title}/{ts}/"

        # Ensure the directory exists (works for local or cloud paths)
        fs, path = fsspec.core.url_to_fs(book_cfg.output_base_path)
        fs.makedirs(path, exist_ok=True)

        eval_book(book_cfg)


if __name__ == "__main__":
    levanter.config.main(main)()
