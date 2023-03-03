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
    # We encode the probabilities as a linear gradient in the Viridis color map
    probs = np.exp(log_probs)

    p10, p90 = np.percentile(probs, [10, 90])
    norm = cm.colors.Normalize(vmin=p10, vmax=p90)

    # Generate the HTML code for the visualization
    html_code = "<div style='font-family: monospace;'>"
    for doc, pdoc, logpdoc in zip(tokens, probs, log_probs):
        for i, token in enumerate(doc):
            prob = pdoc[i]
            lp = logpdoc[i]
            normed = cm.plasma(norm(prob))
            color = (255 * np.array(normed)).astype(int)
            # need this -webkit on Blink and WebKit to let the backround color be transparent
            html_code += (
                f"<span style='background-color: rgba({color[0]}, {color[1]}, {color[2]}, 0.5);' "
                "background-clip: text; -webkit-background-clip: text; color: transparent;"
                f" title='{lp:.2f}'>{token}</span>"
            )
        html_code += "<br>\n"
    html_code += "</div>"

    # Write the HTML code to a file
    with open(output_path, "w") as f:
        f.write(html_code)


# dumb main to test it out
if __name__ == "__main__":
    tokens = [["Hello", "world", "!"], ["This", "is", "a", "test", "."]]
    log_probs = np.random.randn(2, 5)
    visualize_log_probs(tokens, log_probs, "test.html")
