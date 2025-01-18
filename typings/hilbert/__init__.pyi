import numpy as np
from numpy.typing import NDArray

def decode(
    hilberts: NDArray[np.uint64], num_dims: int, num_bits: int
) -> NDArray[np.uint64]: ...
def encode(
    locs: NDArray[np.uint64], num_dims: int, num_bits: int
) -> NDArray[np.uint64]: ...
