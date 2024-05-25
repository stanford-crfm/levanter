from typing import List


def get_olmo_metrics_keys() -> List[str]:
    olmo_metrics = [
        "train",
        "eval/v3-small-dolma_common-crawl-validation",
        "eval/pile",
        "eval/v2-small-4chan-validation",
        "eval/v3-small-wikitext_103-validation",
        "eval/v2-small-m2d2_s2orc-validation",
        "eval/v2-small-pile-validation",
        "eval/v2-small-mc4_en-validation",
        "eval/v3-small-dolma_wiki-validation",
        "eval/v2-small-ice-validation",
        "eval/v2-small-gab-validation",
        "eval/v2-small-ptb-validation",
        "eval/v2-small-m2d2_wiki-validation",
        "eval/v2-small-twitterAEE-validation",
        "eval/v3-small-c4_en-validation",
        "eval/v3-small-dolma_stack-validation",
        "eval/v3-small-dolma_books-validation",
        "eval/v2-small-c4_en-validation",
        "eval/v3-small-m2d2_s2orc-validation",
        "eval/v3-small-dolma_pes2o-validation",
        "eval/v3-small-ice-validation",
        "eval/v3-small-dolma_reddit-validation",
        "eval/v2-small-manosphere-validation",
        "eval/v2-small-c4_100_domains-validation",
        "eval/v2-small-wikitext_103-validation",
    ]
    olmo_keys = []
    for metric in olmo_metrics:
        olmo_keys.extend([
            f"{metric}/CrossEntropyLoss",
            f"{metric}/Perplexity",
        ])
    return olmo_keys


def get_marin_metrics_keys() -> List[str]:
    marin_keys = [
        "train/loss",
        "eval/paloma/falcon-refinedweb/loss",
        "eval/paloma/dolma-v1_5/loss",
        "eval/paloma/gab/loss",
        "eval/paloma/mc4/loss",
        "eval/paloma/m2d2_s2orc_unsplit/loss",
        "eval/paloma/dolma_100_subreddits/loss",
        "eval/paloma/micro_loss",
        "eval/paloma/c4_100_domains/loss",
        "eval/paloma/dolma_100_programing_languages/loss",
        "eval/loss",
        "eval/paloma/redpajama/loss",
        "eval/paloma/m2d2_wikipedia_unsplit/loss",
        "eval/paloma/ptb/loss",
        "eval/paloma/4chan/loss",
        "eval/paloma/wikitext_103/loss",
        "eval/macro_loss",
        "eval/paloma/c4_en/loss",
        "eval/paloma/twitterAAE_HELM_fixed/loss",
        "eval/paloma/manosphere_meta_sep/loss",
        "eval/paloma/macro_loss",
    ]
    return marin_keys


def get_marin_olmo_metrics_mapping():
    mapping = {
        "train/loss": "train/CrossEntropyLoss",
        "eval/paloma/gab/loss": "eval/v2-small-gab-validation/CrossEntropyLoss",
        "eval/paloma/mc4/loss": "eval/v2-small-mc4_en-validation/CrossEntropyLoss",
        "eval/paloma/c4_100_domains/loss": "eval/v2-small-c4_100_domains-validation/CrossEntropyLoss",
        "eval/paloma/ptb/loss": "eval/v2-small-ptb-validation/CrossEntropyLoss",
        "eval/paloma/4chan/loss": "eval/v2-small-4chan-validation/CrossEntropyLoss",
        "eval/paloma/wikitext_103/loss": "eval/v3-small-wikitext_103-validation/CrossEntropyLoss",
        "eval/paloma/wikitext_103/loss": "eval/v2-small-wikitext_103-validation/CrossEntropyLoss",
        "eval/paloma/c4_en/loss": "eval/v3-small-c4_en-validation/CrossEntropyLoss",
        # "eval/paloma/c4_en/loss": "eval/v2-small-c4_en-validation/CrossEntropyLoss",
        "eval/paloma/m2d2_s2orc_unsplit/loss": "eval/v2-small-m2d2_s2orc-validation/CrossEntropyLoss",
        "eval/paloma/dolma_100_subreddits/loss": "eval/v3-small-dolma_reddit-validation/CrossEntropyLoss",
        "eval/paloma/m2d2_wikipedia_unsplit/loss": "eval/v2-small-m2d2_wiki-validation/CrossEntropyLoss",
        "eval/paloma/twitterAAE_HELM_fixed/loss": "eval/v2-small-twitterAEE-validation/CrossEntropyLoss",
        "eval/paloma/manosphere_meta_sep/loss": "eval/v2-small-manosphere-validation/CrossEntropyLoss",
    }
    return mapping


if __name__ == "__main__":
    marin_keys = get_marin_metrics_keys()
    olmo_keys = get_olmo_metrics_keys()
    mapping = get_marin_olmo_metrics_mapping()
    for key in marin_keys:
        if key in mapping:
            continue
        if key not in mapping:
            print(f"Could not find mapping for {key}")