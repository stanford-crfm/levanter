import asyncio
import dataclasses
import functools
import logging
from typing import Optional, Sequence

import equinox as eqx
import jax
import numpy as np
import tensorstore as ts
from jax.random import PRNGKey

import haliax as hax
from haliax import Axis

from levanter.models.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.newdata.dataset import AsyncDataset, MappedAsyncDataset, T_co
from levanter.store.cache import TreeCache
from levanter.store.jagged_array import JaggedArrayStore
from levanter.store.tree_store import TreeStoreBuilder
from levanter.utils.jax_utils import local_cpu_mesh


logger = logging.getLogger(__name__)


class TokenSeqDataset(AsyncDataset[np.ndarray]):
    """
    A dataset that yields sequences of tokens of fixed length from an underlying TreeCache.

    :param doc_cache: the TreeCache to read from
    :param seq_len: The max length of sequences to emit
    """

    def __init__(self, doc_cache: TreeCache[dict], seq_len: int):
        self.doc_cache = doc_cache
        self.seq_len = seq_len
        self._store: Optional[TreeStoreBuilder] = None
        self._cached_len: Optional[int] = None

    async def async_len(self) -> int:
        await self.doc_cache.finished()
        token_arrays = await self._await_token_cache()
        return token_arrays.data_size // self.seq_len

    async def _await_token_cache(self) -> JaggedArrayStore:
        if self._store is None:
            self._store = await self.doc_cache.store_async()
        return self._store.tree["input_ids"]

    async def length_is_known(self) -> bool:
        return await self.doc_cache.length_is_known()

    def is_finite(self) -> bool:
        return True

    async def current_len(self) -> Optional[int]:
        store = await self._await_token_cache()
        return store.data_size // self.seq_len

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        token_arrays = await self._await_token_cache()
        # logger.info(f"Time to get token cache: {time.time() - time_in}")
        len = await self.wait_until_len_at_least(max(indices) + 1)
        if len is not None and len < max(indices) + 1:
            raise ValueError("Requested indices beyond the end of the dataset")
        offsets = np.array(indices) * self.seq_len
        with ts.Batch():
            out = []
            for offset in offsets:
                out.append(token_arrays.data[offset : offset + self.seq_len].read())

        out = await asyncio.gather(*out)

        return out

    def get_batch_sync(self, indices: Sequence[int]) -> Sequence[T_co]:
        token_arrays = self.doc_cache.store.tree["input_ids"]
        # logger.info(f"Time to get token cache: {time.time() - time_in}")
        # len = await self.wait_until_len_at_least(max(indices) + 1)
        # if len is not None and len < max(indices) + 1:
        # raise ValueError("Requested indices beyond the end of the dataset")
        offsets = np.array(indices) * self.seq_len
        with ts.Batch():
            out = []
            for offset in offsets:
                out.append(token_arrays.data[offset : offset + self.seq_len].read())
        # logger.info(f"Time to read token cache: {time.time() - time_in}")

        out = [x.result() for x in out]
        # logger.info(f"Time to wait for token cache: {time.time() - time_in}")
        return out

    async def wait_until_len_at_least(self, length: int) -> int:
        # length is brutally slow to compute, so we cache it
        if self._cached_len is not None and self._cached_len >= length:
            return self._cached_len

        # TODO: would be better to listen for cache updates
        length = await super().wait_until_len_at_least(length)
        self._cached_len = length
        return length


class CausalLmDataset(MappedAsyncDataset[np.ndarray, LmExample]):
    def __init__(
        self,
        dataset: AsyncDataset[np.ndarray],
        QPos: Axis,
        KPos: Axis,
        fcm_prob: float = 0.0,
        key: Optional[PRNGKey] = None,
        ignore_index: Optional[int] = None,
    ):
        self.dataset = dataset
        self.QPos = QPos
        self.KPos = KPos
        self.fcm_prob = fcm_prob
        self.ignore_id = ignore_index

        if self.fcm_prob > 0.0 and self.key is None:
            raise ValueError("must provide key if fcm_prob > 0.0")

        sharding = jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])

        @functools.partial(eqx.filter_jit, out_shardings=sharding)
        def _create_lm_example(tokens, key):
            with local_cpu_mesh():
                tokens = hax.named(tokens, self.QPos)
                example = LmExample.causal(tokens=tokens, ignore_id=self.ignore_id)

                if self.fcm_prob > 0:
                    # masks for attention
                    # We support forgetful causal masking (FCM) which is a technique that improves training speed by
                    # randomly masking out some of the context. This is a bit like dropout, but it's applied to the attention
                    # mask instead of the activations. It's described in https://arxiv.org/abs/2210.13432
                    assert self.key is not None
                    this_key, key = jax.random.split(key)
                    fcm_mask = hax.nn.attention.forgetful_causal_mask(self.KPos, self.fcm_prob, key=this_key)
                    attn_mask = example.attn_mask & AttentionMask.explicit(fcm_mask)
                    example = dataclasses.replace(example, attn_mask=attn_mask)

                return example

        super().__init__(self.dataset, _create_lm_example, key=key)

    async def async_len(self) -> int:
        return await self.dataset.async_len()
