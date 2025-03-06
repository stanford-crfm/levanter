import html
import re
from functools import partial
from typing import Any, List, Optional

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils


def visualize_log_probs(
    tokens: List[List[str]], log_probs: np.ndarray, output_path: str, argmaxes: Optional[List[List[Any]]] = None
):
    """
    Visualizes token log probabilities by embedding data attributes in spans and computing
    their color using JavaScript. Optionally, if argmaxes is provided, each span's title will include
    the argmax value.

    Args:
        tokens: List of list of tokens for each document.
        log_probs: Array of log probabilities, shape (num_docs, num_tokens).
        output_path: Path to write the HTML file to. Fsspec paths are supported.
        argmaxes: Optional list of list of argmax values corresponding to each token.
    """
    html = (
        "<html><head><meta charset='utf-8'><style>"
        ".lp span { font-family: monospace; padding: 0.2em; border-radius: 2px; }"
        "</style></head><body><div class='lp'>"
    )
    for doc_idx, (doc, logpdoc) in enumerate(zip(tokens, log_probs)):
        for i, (token, lp) in enumerate(zip(doc, logpdoc)):
            if argmaxes is not None:
                argmax_val = _escape(argmaxes[doc_idx][i]).replace("'", "&apos;")
                html += f"<span data-lp='{lp:.3f}' data-argmax='{argmax_val}'>{_escape(token)}</span>"
            else:
                html += f"<span data-lp='{lp:.3f}'>{_escape(token)}</span>"
        html += "<br><br>"
    html += (
        "</div>"
        "<script>"
        "const SCALE = 5;\n"
        "document.querySelectorAll('.lp span').forEach(el => {\n"
        "  let lp = parseFloat(el.getAttribute('data-lp'));\n"
        "  clamp_lp = Math.max(lp, -10);\n"
        "  const norm = (clamp_lp + 10) / 10;\n"
        "  const hue = 60 * norm;\n"
        "  const alpha = (1 - Math.exp(lp / SCALE)).toFixed(2);\n"
        "  el.style.background = `hsla(${hue}, 100%, 50%, ${alpha})`;\n"
        "  const argmax = el.getAttribute('data-argmax');\n"
        "  if (argmax !== null) {\n"
        "    el.title = `lp: ${lp.toFixed(3)} | argmax: ${argmax}`;\n"
        "  } else {\n"
        "    el.title = lp.toFixed(3);\n"
        "  }"
        "});"
        "</script>"
        "</body></html>"
    )

    with fsspec.open(output_path, "w") as f:
        f.write(html)


def _escape(s: str) -> str:
    out = html.escape(s, quote=False)
    # we want newlines and other special characters to be visible
    # we also want strings of spaces to be visible
    out = out.replace("\n", "⏎").replace("\t", "⇥")
    out = re.sub(r"  +", lambda m: " " + "␣" * (len(m.group(0)) - 1), out)
    return out


def compute_and_visualize_log_probs(path: str, model, tokenizer, log_prob_fn, test_data, max_docs=128):
    """
    Compute and visualize log probabilities for a given model and dataset.
    Args:
        path:  Path to write the HTML file to. Fsspec paths are supported.
        model:  Model to use for computing log probabilities.
        tokenizer:  Tokenizer to use for decoding tokens.
        log_prob_fn:  Function to compute log probabilities. Can return log probabilities and argmaxes or just log probabilities.
        test_data: Test data to use for computing log probabilities.
        max_docs: Maximum number of documents to visualize.

    Returns:

    """
    log_probs = []
    targets = []
    argmaxes: list = []
    for batch in test_data:
        out = log_prob_fn(model, batch)
        if len(out) == 2:
            b_logprobs, b_argmaxes = out
            log_probs.append(b_logprobs)
            argmaxes.append(b_argmaxes)
        else:
            b_logprobs = out
            log_probs.append(b_logprobs)

        targets.append(batch)

        # TODO: haliax-ify?
        if len(targets) * b_logprobs.shape[0] >= max_docs:
            break
    log_probs = _concatenate(log_probs)
    targets = _concatenate([t.tokens.array for t in targets])
    if argmaxes:
        argmaxes_array = _concatenate(argmaxes)
    else:
        argmaxes_array = None
    # gather the log probs and targets
    # TODO: is this still necessary?
    (targets, log_probs, argmaxes_array) = multihost_utils.process_allgather(
        (targets, log_probs, argmaxes_array), tiled=True
    )
    log_probs = log_probs[:max_docs]
    targets = targets[:max_docs]
    targets = np.array(targets)
    tokens = [_decode_tokens_pretty(tokenizer, t) for t in targets]
    if argmaxes_array is not None:
        argmaxes_array = argmaxes_array[:max_docs]
        argmaxes_array = [_decode_tokens_pretty(tokenizer, a) for a in argmaxes_array]
    log_probs = np.array(log_probs)
    visualize_log_probs(tokens, log_probs, path, argmaxes=argmaxes_array)


def visualize_log_prob_diff(
    tokens: List[List[str]],
    log_probs_a: np.ndarray,
    log_probs_b: np.ndarray,
    output_path: str,
    diff_threshold: float = 5.0,
):
    """
    Visualizes the difference in token log probabilities between two models by embedding
    data attributes in spans and computing their color using JavaScript.

    For each token, the difference (log_probs_a - log_probs_b) is computed and clamped to
    [-diff_threshold, diff_threshold]. Very positive differences appear in blue,
    very negative differences in orange. Near-zero differences are nearly transparent,
    with opacity computed via exponential scaling.

    Args:
        tokens: List of list of tokens for each document.
        log_probs_a: Array of log probabilities from model A, shape (num_docs, num_tokens).
        log_probs_b: Array of log probabilities from model B, shape (num_docs, num_tokens).
        output_path: Path to write the HTML file to. Fsspec paths are supported.
        diff_threshold: Maximum absolute difference to consider; differences beyond this are clamped.
    """
    html = (
        "<html><head><meta charset='utf-8'><style>"
        ".lp span { font-family: monospace; padding: 0.2em; border-radius: 2px; }"
        "</style></head><body><div class='lp'>"
    )
    for doc, lp_a_doc, lp_b_doc in zip(tokens, log_probs_a, log_probs_b):
        for token, lp_a, lp_b in zip(doc, lp_a_doc, lp_b_doc):
            diff = lp_a - lp_b
            html += f"<span data-diff='{diff:.3f}'>{_escape(token)}</span>"
        html += "<br><br>"
    html += (
        "</div>"
        "<script>\n"
        "const SCALE = 3;\n"
        f"const threshold = {diff_threshold};\n"
        "document.querySelectorAll('.lp span').forEach(el => {\n"
        "  let diff = parseFloat(el.getAttribute('data-diff'));\n"
        "  // Clamp the difference\n"
        "  clamp_diff = Math.max(-threshold, Math.min(threshold, diff));\n"
        "  // Compute alpha using exponential scaling so small differences are nearly transparent\n"
        "  const alpha = (1 - Math.exp(-Math.abs(clamp_diff) / SCALE)).toFixed(2);\n"
        "  let color = 'transparent';\n"
        "  if(diff > 0) {\n"
        "    // Positive difference: use blue (e.g., hsla(219,70%,50%,alpha))\n"
        "    color = `hsla(219, 70%, 50%, ${alpha})`;\n"
        "  } else if(diff < 0) {\n"
        "    // Negative difference: use orange (e.g., hsla(39,100%,50%,alpha))\n"
        "    color = `hsla(39, 100%, 50%, ${alpha})`;\n"
        "  }\n"
        "  el.style.background = color;\n"
        "  el.title = `diff: ${diff.toFixed(3)}`;\n"
        "});\n"
        "</script>\n"
        "</body></html>"
    )

    with fsspec.open(output_path, "w") as f:
        f.write(html)


def compute_and_diff_log_probs(path: str, model, comparison_model, tokenizer, log_prob_fn, test_data, max_docs=128):
    """
    Compute and visualize the difference in log probabilities between two models for a given dataset.

    Args:
        path: Path to write the HTML file to. Fsspec paths are supported.
        model: Model to use for computing log probabilities.
        comparison_model: Model to compare against.
        tokenizer: Tokenizer to use for decoding tokens.
        log_prob_fn: Function to compute log probabilities. Can return log probabilities and argmaxes or just log probabilities.
        test_data: Test data to use for computing log probabilities.
        max_docs: Maximum number of documents to visualize.
    """

    log_probs_a = []
    log_probs_b = []
    targets = []
    for batch in test_data:
        targets.append(batch)

        out = log_prob_fn(model, batch)
        if len(out) == 2:
            b_logprobs, b_argmaxes = out
            log_probs_a.append(b_logprobs)
        else:
            b_logprobs = out
            log_probs_a.append(b_logprobs)

        compare_out = log_prob_fn(comparison_model, batch)
        if len(compare_out) == 2:
            b_logprobs, b_argmaxes = compare_out
            log_probs_b.append(b_logprobs)
        else:
            b_logprobs = compare_out
            log_probs_b.append(b_logprobs)

        if len(targets) * b_logprobs.shape[0] >= max_docs:
            break

    log_probs_a = _concatenate(log_probs_a)
    log_probs_b = _concatenate(log_probs_b)
    log_probs_a = np.array(log_probs_a[:max_docs])
    log_probs_b = np.array(log_probs_b[:max_docs])

    targets = _concatenate([t.tokens.array for t in targets])
    tokens = [_decode_tokens_pretty(tokenizer, t) for t in targets]
    visualize_log_prob_diff(tokens, log_probs_a, log_probs_b, path)


@partial(jax.jit, out_shardings=None)
def _concatenate(x):
    return jnp.concatenate(x, axis=0)


def _decode_tokens_pretty(tok, ids):
    return [
        tok.convert_tokens_to_string([x]) if x is not None else tok.unk_token for x in tok.convert_ids_to_tokens(ids)
    ]


# dumb main to test it out
if __name__ == "__main__":
    np.random.seed(1)
    tokens = [["Hello", "world", "!"], ["This", "is", "a", "test", "."]]
    log_probs = np.log(np.random.uniform(size=(2, 5)))
    visualize_log_probs(tokens, log_probs, "test.html")

    # test diff
    log_probs_a = np.log(np.random.uniform(size=(2, 5)))
    log_probs_b = np.log(np.random.uniform(size=(2, 5)))
    visualize_log_prob_diff(tokens, log_probs_a, log_probs_b, "test_diff.html")
