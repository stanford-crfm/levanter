import json
import tempfile
import os

import datasets
import huggingface_hub

# dataset_weights = {
#     "joelito/eurlex_resources": 0.2,
#     "pile-of-law/pile-of-law": 0.55,
#     "joelito/Multi_Legal_Pile": 0.10,
#     "joelito/mc4_legal": 0.15,
# }
#
# # first, determine the urls the datasets use
# urls = {}
# def mk_url(repo_name, filename):
#     return f"https://huggingface.co/datasets/{repo_name}/resolve/main/{filename}"
#
# with tempfile.TemporaryDirectory() as tmp_dir:
#     cwd = os.getcwd()
#     os.chdir(tmp_dir)
#     for dataset_name, weight in dataset_weights.items():
#         my_urls = {"train": [], "validation": []}
#         repo = huggingface_hub.Repository(dataset_name, clone_from=f"https://huggingface.co/datasets/{dataset_name}", skip_lfs_files=True)
#         # walk through the files in the repo, look for .jsonl.* files
#         for root, dirs, files in os.walk(repo.local_dir):
#             for file in files:
#                 rel_path = os.path.relpath(os.path.join(root, file), repo.local_dir)
#                 is_train = "train" in file
#                 is_validation = "validation" in file or "valid" in file
#                 if ".jsonl." in file:
#                     if is_validation:
#                         my_urls["validation"].append(mk_url(dataset_name, rel_path))
#                     else:
#                         if not is_train:
#                             print(f"Found file {file} that is neither train nor validation. Guessing train")
#                         my_urls["train"].append(mk_url(dataset_name, rel_path))
#
#         urls[dataset_name] = my_urls
#
#     os.chdir(cwd)


# now wikipedia
wiki_langs = [
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "ga",
    "hr",
    "hu",
    "it",
    "lt",
    "lv",
    "mt",
    "nl",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "sv",
]

for lang in wiki_langs:
    # dataset = datasets.load_dataset("olm/wikipedia", language=lang, date="20221101")
    dataset = datasets.load_dataset("wiki40b", name=lang, beam_runner="DirectRunner")

    # write a sample
    with open(f"wiki_{lang}_sample.jsonl", "w") as f:
        for i in range(100):
            f.write(json.dumps(dataset["train"][i]))
