# Hilbert Bytes

Hilbert Bytes is a python library for converting to and from points in
d-dimensions and their corresponding index on a hilbert curve. It's similar to
[hilbertcurve](https://pypi.org/project/hilbertcurve/) and
[numpy-hilbert-curve](https://pypi.org/project/numpy-hilbert-curve/) but is
faster and more space efficient than either by keeping manipulations at the byte
level, and using numba to compile the results. It also uses arbitrary precision
integers, allowing you to make the grid arbitrarily fine

## Installation

```sh
pip install hilbert-bytes
```

## Usage

```py
import hilbert_bytes
import numpy as np

points = ... # arbitrary d-dimensional points
num, dim = points.shape
# convert to big-endian bytes
points_bytes = points[..., None].astype(">u8").view("u1")
index_bytes = hilbert_bytes.encode(points_bytes)  # indies as big-endian ints
new_points_bytes = hilbert_bytes.decode(index_bytes)
```

If you want the indices as multi-byte ints, you can can do a similar trick in reverse:

```py
index_bytes = ... # an array of big-endian ints
indices = index_bytes.view(">u8").astype("u8")[..., 0]
```

But note that this will only work if your index fits in 8 bytes
