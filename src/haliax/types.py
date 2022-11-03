from dataclasses import dataclass
from typing import Sequence, Tuple, Union

from jax.lax import Precision


@dataclass(frozen=True)
class Axis:
    name: str
    size: int

    def alias(self, new_name: str):
        return Axis(new_name, self.size)


AxisSpec = Union[Axis, Sequence[Axis]]
Scalar = Union[float, int]

PrecisionLike = Union[None, str, Precision, Tuple[str, str], Tuple[Precision, Precision]]
