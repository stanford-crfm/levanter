import html
from typing import List

import numpy as np
from matplotlib import cm


def visualize_log_probs(tokens: List[List[str]], log_probs: np.ndarray, output_path: str):
    """
    Visualizes the log probabilities of tokens output by a language model as a heatmap.

    Args:
    - tokens (List[List[str]]): A list of lists of tokens, where each sublist represents a document.
    - log_probs (np.ndarray): An array of shape (n_docs, max_seq_len) containing the log probabilities
                              of each token in each document.
    - output_path (str): The path to the output HTML file.
    """
    probs = np.exp(log_probs)

    # We encode the log probabilities as a linear gradient
    norm = cm.colors.Normalize(vmin=-10.0, vmax=0.0)

    # Generate the HTML code for the visualization
    # css preamble to define a style for the span elements
    css_preamble = """
    <style>
    .logprobs span {
        background-clip: text;
        -webkit-background-clip: text;
    }
    </style>
    """

    html_code = f"<html>{css_preamble}<div class='logprobs' style='font-family: monospace;'>"
    for doc, pdoc, logpdoc in zip(tokens, probs, log_probs):
        for i, token in enumerate(doc):
            prob = float(pdoc[i])
            lp = float(logpdoc[i])
            normed = cm.plasma(norm(lp))
            color = (255 * np.array(normed)).astype(int)
            html_code += (
                f"<span style='background: rgba({color[0]}, {color[1]}, {color[2]}, {min(0.5,1-prob):.2f});' "
                f" title='{lp:.4f}'>{_escape(token)}</span>"
            )
        html_code += "<br>\n"
    html_code += "</div></html>"

    # Write the HTML code to a file
    with open(output_path, "w") as f:
        f.write(html_code)


def _escape(s: str) -> str:
    return html.escape(s, quote=False)


# dumb main to test it out
if __name__ == "__main__":
    tokens = [["Hello", "world", "!"], ["This", "is", "a", "test", "."]]
    log_probs = np.log(np.random.uniform(size=(2, 5)))
    visualize_log_probs(tokens, log_probs, "test.html")
