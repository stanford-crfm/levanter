# cf https://github.com/lucidrains/flash-attention-jax
# cf https://tridao.me/publications/flash2/flash2.pdf
# cf https://arxiv.org/pdf/2205.14135.pdf
import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

import haliax as hax


# TODO: tune
BLOCK_SIZE = 128


def flash_attention(
    QPos: hax.AxisSelector,
    KPos: hax.AxisSelector,
    Key: hax.AxisSelector,
    q: hax.NamedArray,
    k: hax.NamedArray,
    v: hax.NamedArray,
    mask: Optional[hax.NamedArray] = None,
    dropout: float = 0.0,
    *,
    inference: bool,
    key: Optional[PRNGKeyArray] = None,
):
    """
    Flash Attention impl, vaguely following the v2 paper.

    Args:
        Key: axis of key dim.
    """
    if not inference and dropout > 0 and key is None:
        raise ValueError("key must be provided for training")

    if dropout < 0 or dropout > 1:
        raise ValueError(f"invalid dropout {dropout}")

    # premultiply by 1/sqrt(d_k) for normal dot product attention
    q = q * jax.lax.rsqrt(float(q.axis_size(Key)))

    QPos = q.resolve_axis(QPos)
    KPos = k.resolve_axis(KPos)

    if QPos.size % BLOCK_SIZE != 0:
        raise ValueError(f"q axis size {q.axis_size(QPos)} is not a multiple of {BLOCK_SIZE}")
    if KPos.size % BLOCK_SIZE != 0:
        raise ValueError(f"k axis size {k.axis_size(KPos)} is not a multiple of {BLOCK_SIZE}")

    QPosBlock = QPos.resize(BLOCK_SIZE)  # Br in the paper
    KPosBlock = KPos.resize(BLOCK_SIZE)  # Bc in the paper

    # number of blocks for Q and K
    Tr = hax.Axis("Tr", QPos.size // BLOCK_SIZE)
    Tc = hax.Axis("Tc", KPos.size // BLOCK_SIZE)

    if mask is not None:
        mask = mask.broadcast_to((QPos, KPos))  # make sure mask is broadcastable

    def do_o_block(i):
        # Step 1: Divide Q into 𝑇𝑟 = \ceil(𝑁/Br) blocks of size Br x d each,
        q_i = q.slice(QPos, QPosBlock, i * BLOCK_SIZE)

        # Step 2: init O_i = 0, sumexp_i = 0, max_i = -inf
        o_i = 0.0 * q_i  # unfortunately zeros_like doesn't work super well
        sumexp_i = hax.zeros((QPosBlock,))
        max_i = hax.full((QPosBlock,), -jnp.inf)

        do_qk_block_i = functools.partial(do_qk_block, q_i, i)
        o_i, sumexp_i, max_i = hax.fold(do_qk_block_i, Tc)((o_i, sumexp_i, max_i), jnp.arange(Tc.size))

        # Step 13: compute L_i = m_i^{Tc} + log(\ell_i^{Tc})
        o_i = o_i / sumexp_i
        # Step 12: compute O_i = diag(\ell_i^{Tc})^{-1} O_i^{Tc}
        L_i = max_i + hax.log(sumexp_i)

        return o_i, L_i

    def do_qk_block(q_i, i, carry, j):
        # Step 1: Divide Q into 𝑇𝑟 = \ceil(𝑁/Br) blocks of size Br x d each,
        #         K and V into 𝑇𝑐 = \ceil(𝑁/Bc) blocks of size Bc x d each.
        o_i, sumexp_i, old_max_i = carry
        k_j = k.slice(KPos, KPosBlock, j * BLOCK_SIZE)
        v_j = v.slice(KPos, KPosBlock, j * BLOCK_SIZE)

        # TODO: precision
        # Step 8: compute Sij = QiKj^T
        attn_ij = hax.dot(Key, q_i, k_j)

        if mask is not None:
            mask_ij = mask.slice(QPos, QPosBlock, i * BLOCK_SIZE).slice(KPos, KPosBlock, j * BLOCK_SIZE)
            attn_ij = hax.where(mask_ij, attn_ij, -1e10)

        # TODO: causal
        # TODO: dropout

        max_i = hax.maximum(old_max_i, hax.max(attn_ij, axis=KPosBlock))
        P_ij = hax.exp(attn_ij - max_i)

        exp_diff = hax.exp(old_max_i - max_i)
        sumexp_i = exp_diff * sumexp_i + hax.sum(P_ij, axis=KPosBlock)
        o_i = exp_diff * o_i + hax.dot(KPosBlock, P_ij, v_j)

        return (o_i, sumexp_i, max_i)

    o, ell = hax.map(do_o_block, Tr)(jnp.arange(Tr.size))

    o = o.flatten_axes((Tr, QPosBlock), QPos)

    return o
