#import os
#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
import haliax as hax
from jax import random
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from levanter.models.attention import AttentionMask
from haliax.partitioning import ResourceAxis, ResourceMapping
from levanter.utils.jax_utils import create_fsdp_mesh
from levanter.models.lm_model import LmExample
from tqdm import tqdm
from levanter.tracker.histograms import sharded_histogram

def _get_llama_config(use_flash=False, num_kv_heads=4, seq_len=1024) -> LlamaConfig:
    return LlamaConfig(
        seq_len=seq_len,
        hidden_dim=64,
        num_layers=8,
        num_heads=16,
        num_kv_heads=num_kv_heads,
        rope_scaling=None,
        gradient_checkpointing=False,  # disable for tests so debugging is easier
        use_flash_attention=use_flash,
        flash_attention_block_size=8 if use_flash else None,
        measure_act_stats=True,
    )

def setup():
    llama_config = _get_llama_config()
    Batch = hax.Axis("batch", 16)
    Vocab = hax.Axis("vocab", 512)
    Pos = llama_config.Pos
    input_ids = hax.random.randint(random.PRNGKey(0), (Batch, Pos), 0, Vocab.size)
    loss_mask = 1 - hax.nn.one_hot(-1, Pos, dtype=jax.numpy.float32)
    ex = LmExample(tokens=input_ids, loss_mask=loss_mask, attn_mask=AttentionMask.causal())

    llama_model = LlamaLMHeadModel.init(Vocab=Vocab, config=llama_config, key=random.PRNGKey(0))
    return llama_model, ex


def main():
    mesh = create_fsdp_mesh(1, jax.device_count(), 1)
    with mesh:
        model, ex = setup()
        with hax.axis_mapping({"batch": ResourceAxis.DATA, "embed": ResourceAxis.MODEL}):
            model = hax.shard(model)
            ex = hax.shard(ex)
            @hax.named_jit
            def forward(ex):
                return model.compute_loss(ex)

            test = forward(ex)
            with jax.profiler.trace("./trace", create_perfetto_trace=True, create_perfetto_link=False):
                for i in tqdm(range(3)):
                    res = forward(ex)
                    print(res)

            #Batch = hax.Axis("batch", 1024)
            #Mlp = hax.Axis("mlp", 524288)
            #inputs = hax.random.normal(random.PRNGKey(0), (Batch, Mlp))
            #inputs = hax.shard(inputs)
            #bins = jax.numpy.linspace(-1, 1, 100)

            #with jax.profiler.trace("./trace", create_perfetto_trace=True, create_perfetto_link=False):
            #    for i in range(3):
            #        res = sharded_histogram(inputs.array + i, bins)

if __name__ == "__main__":
    main()