import datasets


try:
    dsQ = datasets.load_from_disk("instruction_tuning")
except:  # noqa: E722
    ds1 = datasets.load_dataset("Muennighoff/P3", split="train", streaming=True)
    ds2 = datasets.load_dataset("Muennighoff/natural-instructions", split="train", streaming=True)
    print("map")
    ds2 = ds2.map(lambda x: {"inputs": "\n".join([x["definition"], x["input"]]), "targets": x["targets"]})
    print("interleave")

    dsQ = datasets.interleave_datasets([ds1, ds2])
    print("shuffle")
    dsQ = dsQ.shuffle(seed=42, buffer_size=10000)
    print("save")

    dsQ.save_to_disk("instruction_tuning")
    print("done")


# TODO: export
