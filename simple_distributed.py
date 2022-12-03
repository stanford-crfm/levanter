import jax
import transformers

jax.distributed.initialize(local_device_ids=list(range(0, 4)))

print(jax.devices())
print(jax.process_count())
