import json

import datasets
import gcsfs


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


dataset_weights = {
    "joelito/eurlex_resources": 0.2,
    "pile-of-law/pile-of-law": 0.35,
    "joelito/Multi_Legal_Pile": 0.10,
    "joelito/mc4_legal": 0.15,
    "olm/wikipedia": 0.2,
}

# wikipedia will be weighted proportionally to the number of articles in each language

# all_datasets = {
#     k: datasets.load_dataset(k) for k in dataset_weights.keys() if k != "olm/wikipedia"
# }

all_datasets = {}
all_datasets["joelito/eurlex_resources"] = datasets.load_dataset("joelito/eurlex_resources", "all_all", streaming=True)
all_datasets["pile-of-law/pile-of-law"] = datasets.load_dataset("pile-of-law/pile-of-law", "all", streaming=True)
all_datasets["joelito/Multi_Legal_Pile"] = datasets.load_dataset("joelito/Multi_Legal_Pile", "all_all", streaming=True)
# datasets["joelito/mc4_legal"] = datasets.load_dataset("joelito/mc4_legal", "all_all")


# wikipedia is special because it's a single dataset with multiple languages
wiki_datasets = []
for lang in wiki_langs:
    wiki_datasets.append(datasets.load_dataset("olm/wikipedia", language=lang, date="20221101"))

all_datasets["olm/wikipedia"] = datasets.concatenate_datasets(wiki_datasets)

# now we need to weight each dataset. we can use datasets interleave
probabilities = [dataset_weights[k] for k in dataset_weights.keys()]
# combined = datasets.interleave_datasets(list(all_datasets.values()), seed=41, probabilities=probabilities, stopping_strategy="all_exhausted")
combined = datasets.concatenate_datasets(list(all_datasets.values()))
combined = combined.shuffle(seed=42)

fs = gcsfs.GCSFileSystem()

combined.save_to_disk("gs://levanter-data/legal/v1", fs=fs)


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


for lang in wiki_langs:
    dataset = datasets.load_dataset("olm/wikipedia", language=lang, date="20221101")

    # write a sample
    with open(f"wiki_{lang}_sample.jsonl", "w") as f:
        for i in range(100):
            f.write(json.dumps(dataset["train"][i]))
