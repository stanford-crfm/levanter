from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional

import levanter
from levanter.main.eval_careless_lm import EvalCarelessLmConfig, main as eval_book


@dataclass
class BookConfig:
    """Overrides for a single book evaluation."""

    txt_path: str
    book_title: Optional[str] = None
    plot_path: Optional[str] = None
    histogram_path: Optional[str] = None
    pz_data_path: Optional[str] = None
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

        # Fill in book title if not provided
        if not book_cfg.book_title:
            book_cfg.book_title = name

        # Set per-book output directory under base path
        base_path = cfg.base_eval.output_base_path.rstrip("/")
        book_cfg.output_base_path = f"{base_path}/{book_cfg.book_title}/"

        eval_book(book_cfg)


if __name__ == "__main__":
    levanter.config.main(main)()
