from levanter.store import JaggedArrayStore

a = JaggedArrayStore.open("gs://marin-us-central2/tokenized/dolma/algebraic-stack-cc00cf/train/input_ids", dtype=int)

a.data_size

150,849,275