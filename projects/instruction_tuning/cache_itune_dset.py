import datasets


try:
    dsQ = datasets.load_from_disk("instruction_tuning")
except:  # noqa: E722
    ds1 = datasets.load_dataset("Muennighoff/P3", split="train")
    ds1.save_to_disk("instruction_tuning_p3")
    ds2 = datasets.load_dataset("Muennighoff/natural-instructions", split="train")
    print("map")
    ds2 = ds2.map(lambda x: {"inputs": "\n".join([x["definition"], x["inputs"]]), "targets": x["targets"]})
    print("interleave")

    dsQ = datasets.interleave_datasets([ds1, ds2])
    print("shuffle")
    dsQ = dsQ.shuffle(seed=42)
    print("save")

    dsQ.save_to_disk("instruction_tuning")
    print("done")


# TODO: export
