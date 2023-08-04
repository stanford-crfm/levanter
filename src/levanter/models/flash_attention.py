# cf https://github.com/lucidrains/flash-attention-jax
# cf https://tridao.me/publications/flash2/flash2.pdf
# cf https://arxiv.org/pdf/2205.14135.pdf
from typing import Optional, Tuple

import equinox
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

    return _flash_attention((q, k, v), QPos, KPos, Key, mask, dropout, inference=inference, key=key)


@equinox.filter_custom_vjp
def _flash_attention(
    qkv: Tuple[hax.NamedArray, hax.NamedArray, hax.NamedArray],
    QPos: hax.Axis,
    KPos: hax.Axis,
    Key: hax.Axis,
    mask: Optional[hax.NamedArray] = None,
    dropout: float = 0.0,
    *,
    inference: bool,
    key: Optional[PRNGKeyArray] = None,
) -> hax.NamedArray:
    return _flash_attention_forward(qkv, QPos, KPos, Key, mask, dropout, inference=inference, key=key)[0]


def _flash_attention_forward(
    qkv,
    QPos: hax.AxisSelector,
    KPos: hax.AxisSelector,
    Key: hax.AxisSelector,
    mask: Optional[hax.NamedArray] = None,
    dropout: float = 0.0,
    *,
    inference: bool,
    key: Optional[PRNGKeyArray] = None,
):
    q, k, v = qkv
    if QPos.size % BLOCK_SIZE != 0:
        raise ValueError(f"q axis size {q.axis_size(QPos)} is not a multiple of {BLOCK_SIZE}")
    if KPos.size % BLOCK_SIZE != 0:
        raise ValueError(f"k axis size {k.axis_size(KPos)} is not a multiple of {BLOCK_SIZE}")

    QPosBlock = QPos.resize(BLOCK_SIZE)  # Br in the paper
    KPosBlock = KPos.resize(BLOCK_SIZE)  # Bc in the paper

    q_batch_axes: Tuple[hax.Axis, ...] = hax.eliminate_axes(q.axes, (QPos, Key))

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

        sumexp_i = hax.zeros(q_batch_axes + (QPosBlock,), q.dtype)
        max_i = hax.full(q_batch_axes + (QPosBlock,), -jnp.inf)

        def do_qk_block(carry, j):  # computes softmax(Q_i K_j^T) V_j
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

            # TODO: block causal
            # TODO: dropout

            # Step 9: Compute m_i^j = max(m_i^{j-1}, rowmax(S_i^j)), P_i^j = exp(S_i^j - m_i^j),
            # ...    l_i^j = exp(m_i^{j-1} - max(S_i^j)) + rowsum(P_i^j)
            max_i = hax.maximum(old_max_i, hax.max(attn_ij, axis=KPosBlock))
            P_ij = hax.exp(attn_ij - max_i)

            exp_diff = hax.exp(old_max_i - max_i)
            sumexp_i = exp_diff * sumexp_i + hax.sum(P_ij, axis=KPosBlock)

            # Step 10: Compute O_i = diag(exp(m_i^{j-1} - m_i^j) O_i + P_i^j V_j
            o_i = exp_diff * o_i + hax.dot(KPosBlock, P_ij, v_j)

            return (o_i, sumexp_i, max_i)

        o_i, sumexp_i, max_i = hax.fold(do_qk_block, Tc)((o_i, sumexp_i, max_i), jnp.arange(Tc.size))

        # Step 12: compute O_i = diag(\ell_i^{Tc})^{-1} O_i^{Tc}
        o_i = o_i / sumexp_i
        # Step 13: compute L_i = m_i^{Tc} + log(\ell_i^{Tc})
        L_i = max_i + hax.log(sumexp_i)

        return o_i, L_i

    o, ell = hax.map(do_o_block, Tr)(jnp.arange(Tr.size))

    # TODO: reorder axes so it works properly with batching
    o = o.flatten_axes((Tr, QPosBlock), QPos)
    ell = ell.flatten_axes((Tr, QPosBlock), QPos)

    return o, (o, ell)


def _flash_attention_backward(
    residuals,
    grad_in: hax.NamedArray,
    qkv,
    QPos: hax.AxisSelector,
    KPos: hax.AxisSelector,
    Key: hax.AxisSelector,
    mask: Optional[hax.NamedArray] = None,
    dropout: float = 0.0,
    *,
    inference: bool,
    key: Optional[PRNGKeyArray] = None,
):
    O, L = residuals
    q, k, v = qkv
    dO = grad_in

    Tr = hax.Axis("Tr", QPos.size // BLOCK_SIZE)
    Tc = hax.Axis("Tc", KPos.size // BLOCK_SIZE)

    if mask is not None:
        mask = mask.broadcast_to((QPos, KPos))  # make sure mask is broadcastable

    KPosBlock = KPos.resize(BLOCK_SIZE)
    QPosBlock = QPos.resize(BLOCK_SIZE)

    # Compute D = rowsum(dO * O), write D to HBM and divide it into Tr blocks of size Br each.
    # in the FA2 paper D is said to be \in R^{d}, but that doens't maske sense.
    # Triton impl has it as R^{QPos}, which makes more sense.
    D = hax.sum(dO * O, axis=Key)

    def do_kv_block(dQ, j):
        k_j = k.slice(KPos, KPosBlock, j * BLOCK_SIZE)
        v_j = v.slice(KPos, KPosBlock, j * BLOCK_SIZE)

        def do_inner_block(accum, i):
            dQ, dK_j, dV_j = accum
            q_i = q.slice(QPos, QPosBlock, i * BLOCK_SIZE)
            # the FA2 paper says to read in this o_i, but it's not used anywhere. I think it's copypasta from FA1.
            # o_i = O.slice(QPos, QPosBlock, i * BLOCK_SIZE)

            dQ_i = dQ.slice(QPos, QPosBlock, i * BLOCK_SIZE)
            dO_i = dO.slice(QPos, QPosBlock, i * BLOCK_SIZE)
            L_i = L.slice(QPos, QPosBlock, i * BLOCK_SIZE)
            D_i = D.slice(QPos, QPosBlock, i * BLOCK_SIZE)

            # TODO: precision
            attn_ij = hax.dot(Key, q_i, k_j)

            if mask is not None:
                mask_ij = mask.slice({QPos: i * BLOCK_SIZE, KPos: j * BLOCK_SIZE}, {QPos: QPosBlock, KPos: KPosBlock})
                attn_ij = hax.where(mask_ij, attn_ij, -1e10)

            p_ij = hax.exp(attn_ij - L_i)
            dV_j = dV_j + hax.dot(QPosBlock, p_ij, dO_i)
            dP_ij = hax.dot(Key, dO_i, v_j)
            dAttn_ij = p_ij * (dP_ij - D_i)

            dQ_i = dQ_i + hax.dot(KPosBlock, dAttn_ij, k_j)
            # dQ[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE] = dQi
            dQ = dQ.updated_slice({QPos: i * BLOCK_SIZE}, dQ_i)
            dK_j = dK_j + hax.dot(QPosBlock, dAttn_ij, q_i)

            return dQ, dK_j, dV_j

        dK_j = k_j * 0.0
        dV_j = v_j * 0.0

        dQ, dK_j, dV_j = hax.fold(do_inner_block, Tr)((dQ, dK_j, dV_j), jnp.arange(Tr.size))

        return dQ, (dK_j, dV_j)

    dQ = q * 0.0
    dQ, (dK, dV) = hax.scan(do_kv_block, Tc)(dQ, jnp.arange(Tc.size))

    # dQ is already the right shape because it's folded over rather than scanned over
    dK = dK.flatten_axes((Tc, KPosBlock), KPos)
    dV = dV.flatten_axes((Tc, KPosBlock), KPos)

    return dQ, dK, dV


_flash_attention.defvjp(_flash_attention_forward, _flash_attention_backward)
