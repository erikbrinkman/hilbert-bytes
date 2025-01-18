import numba as nb
import numpy as np
from numpy.typing import NDArray

# NOTE in most of the numba code, update assignments, e.g. += or ^= fail, so
# you'll see a lot of x = x + y


@nb.njit(nb.uint8[:, :](nb.uint8[:, :], nb.uint64))  # type: ignore
def _right_shift(binary: NDArray[np.uint8], k: int) -> NDArray[np.uint8]:
    nbytes = k // 8
    if nbytes > 0:
        new_binary = np.empty_like(binary)
        new_binary[:, :nbytes] = 0
        new_binary[:, nbytes:] = binary[:, :-nbytes]
        binary = new_binary

    nbits = k % 8
    if nbits > 0:
        new_binary = (binary >> nbits).astype("u1")
        new_binary[:, 1:] = new_binary[:, 1:] + (binary[:, :-1] << (8 - nbits))
        binary = new_binary
    return binary


@nb.njit(nb.uint8[:, ::1](nb.uint8[:, :]))
def _gray_encode(arr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return arr ^ _right_shift(arr, 1)


@nb.njit(nb.uint8[:, ::1](nb.uint8[:, ::1]))
def _gray_decode(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    # Loop the log2(bits) number of times necessary, with shift and xor.
    _, nbytes = gray.shape
    shift: int = (1 << np.uint64(np.log2(nbytes - 1))) * 8  # type: ignore
    while shift > 0:
        gray[:] = gray ^ _right_shift(gray, shift)
        shift >>= 1
    return gray


@nb.njit(nb.void(nb.uint8[:], nb.uint8[:]))
def _transpose_bits(inp: NDArray[np.uint8], out: NDArray[np.uint8]) -> None:
    """tranpose the bits in one bit aray into another bit array"""
    (nbytes,) = inp.shape
    mask = np.uint8(128)
    target: int = 0
    for byte in inp:
        for _ in range(8):
            out[target] <<= 1
            if byte & mask:
                out[target] += 1
            byte <<= 1
            target += 1
            target %= nbytes


@nb.njit(nb.uint8[:, ::1](nb.uint8[:, :]))
def _transpose_bits_broadcasted(inp: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """the same as transpose bits, but broadcasted over the leading dimension"""
    num, nbytes = inp.shape
    out = np.zeros((num, nbytes), "u1")
    for i in nb.prange(num):
        _transpose_bits(inp[i], out[i])
    return out


@nb.njit(nb.void(nb.uint8[:], nb.uint8[:]))
def _inv_transpose_bits(inp: NDArray[np.uint8], out: NDArray[np.uint8]) -> None:
    (nbytes,) = inp.shape
    mask = np.uint8(128)
    source: int = 0
    for ind in range(nbytes):
        for _ in range(8):
            out[ind] <<= 1
            if inp[source] & mask:
                out[ind] += 1
            inp[source] <<= 1
            source += 1
            source %= nbytes


@nb.njit(nb.uint8[:, ::1](nb.uint8[:, :]))
def _inv_transpose_bits_broadcasted(inp: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """the same as transpose bits, but broadcasted over the leading dimension"""
    num, nbytes = inp.shape
    out = np.zeros((num, nbytes), "u1")
    for i in nb.prange(num):
        _inv_transpose_bits(inp[i], out[i])
    return out


@nb.njit(nb.uint8[:, ::1](nb.uint8[:, :, :]))
def encode(points: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Encode d-dimensional points into their indices on a hilbert curve.

    This function takes points in a d-dimensional space, and converts them to
    their index on the hilbert curve. All numbers are represented as arbitrary
    precision integers in big-endian form.

    Example
    -------
    If you want to use this native multi-byte integers, you can first cast them
    to a big-endian variant, then view it as bytes.

    ::

        points = ...
        point_bytes = points[..., None].astype(">u8").view("u1")
        res = hilbert_bytes.encode(point_bytes)

    Parameters
    ----------
    points : (n, d, p)
        A collection of n, d-dimensional points stored as p-byte big-endian
        unsigned integers.

    Returns
    -------
    indices : (n, dp)
        A collection of n big-endian unsigned integers that correspond to the
        index along the hilbert-curve for the input points.
    """
    num, ndim, nbytes = points.shape

    # iterate forwards through the bytes and bits
    for byte in range(nbytes):
        for bit in range(8):
            bitmask = 1 << (7 - bit)
            # iterate forwards through the dimensions.
            for dim in range(ndim):
                # identify which ones have this bit active
                mask = points[..., dim, byte] & bitmask
                # this is fully broadcast across all bits
                full_mask = np.where(mask, np.uint8(255), np.uint8(0))
                # this is only broadcast through the lower bits
                partial_mask = (mask - 1) & full_mask
                not_partial_mask = bitmask - 1 - partial_mask

                # where this bit is on, invert the 0 dimension for lower bits
                points[:, 0, byte + 1 :] = points[:, 0, byte + 1 :] ^ full_mask[:, None]
                points[:, 0, byte] = points[:, 0, byte] ^ partial_mask

                # where the bit is off, exchange the lower bits with the 0 dimension
                # first invert the bytes
                to_flip = ~full_mask[:, None] & (
                    points[:, 0, byte + 1 :] ^ points[:, dim, byte + 1 :]
                )
                points[:, dim, byte + 1 :] = points[:, dim, byte + 1 :] ^ to_flip
                points[:, 0, byte + 1 :] = points[:, 0, byte + 1 :] ^ to_flip

                # then the bits
                to_flip = not_partial_mask & (points[:, 0, byte] ^ points[:, dim, byte])
                points[:, dim, byte] = points[:, dim, byte] ^ to_flip
                points[:, 0, byte] = points[:, 0, byte] ^ to_flip

    # now transpose
    byte_transposed = np.swapaxes(points, 1, 2).copy().reshape((num * nbytes, ndim))
    bit_transposed = _inv_transpose_bits_broadcasted(byte_transposed)

    # decode grey encoding
    return _gray_decode(bit_transposed.reshape((num, nbytes * ndim)))


@nb.njit(nb.uint8[:, ::1, :](nb.uint8[:, :], nb.int64))
def decode(indices: NDArray[np.uint8], ndim: int) -> NDArray[np.uint8]:
    """Decode dp-dimensional indices into d-dimensional points.

    This function takes indices on the hilbert curve, and the output dimension
    and converts them to their coresponding points. All numbers are represented
    as arbitrary precision integers in big-endian form.

    Example
    -------
    If you want to use this native multi-byte integers, you can first cast them
    to a big-endian variant, then view it as bytes.

    ::

        indices = ...
        index_bytes = indices[..., None].astype(">u8").view("u1")
        res = hilbert_bytes.decode(index_bytes, 2)

    Parameters
    ----------
    indices : (n, dp)
        A collection of n indices stored as dp-byte big-endian unsigned
        integers.
    ndim : d
        The dimension of points to decode into. It must divide dp, but you can
        always zero pad the left of indices.

    Returns
    -------
    points : (n, d, p)
        A collection of n d-dimensional points that correspond to the indices
        along the hilbert-curve.
    """
    num, ntotbytes = indices.shape
    if ntotbytes % ndim != 0:
        raise ValueError(
            f"num_dims ({ndim}) must evenly divide byte dimension ({ntotbytes})"
        )
    nbytes = ntotbytes // ndim

    # gray-code the bytes
    init_gray = _gray_encode(indices)

    # transpose the bits
    transposed = _transpose_bits_broadcasted(init_gray.reshape((num * nbytes, ndim)))
    gray = np.swapaxes(transposed.reshape((num, nbytes, ndim)), 1, 2)

    # iterate backward through whole bytes
    for byte in range(nbytes - 1, -1, -1):
        # iterate through the bits
        for bit in range(7, -1, -1):
            bitmask = 1 << (7 - bit)
            # Iterate backwards through the dimensions.
            for dim in range(ndim - 1, -1, -1):
                # Identify which ones have this bit active.
                mask = gray[:, dim, byte] & bitmask
                # this is fully broadcast across all bits
                full_mask = np.where(mask, np.uint8(255), np.uint8(0))
                # this is only broadcast through the lower bits
                partial_mask = (mask - 1) & full_mask
                not_partial_mask = bitmask - 1 - partial_mask

                # where this bit is on, invert the 0 dimension for lower bits.
                gray[:, 0, byte + 1 :] = gray[:, 0, byte + 1 :] ^ full_mask[:, None]
                gray[:, 0, byte] = gray[:, 0, byte] ^ partial_mask

                # where the bit is off, exchange the lower bits with the 0 dimension.
                # first do the full bytes
                to_flip = ~full_mask[:, None] & (
                    gray[:, 0, byte + 1 :] ^ gray[:, dim, byte + 1 :]
                )
                gray[:, dim, byte + 1 :] = gray[:, dim, byte + 1 :] ^ to_flip
                gray[:, 0, byte + 1 :] = gray[:, 0, byte + 1 :] ^ to_flip

                # then do the partial bits
                to_flip = not_partial_mask & (gray[:, 0, byte] ^ gray[..., dim, byte])
                gray[:, dim, byte] = gray[:, dim, byte] ^ to_flip
                gray[:, 0, byte] = gray[:, 0, byte] ^ to_flip

    return gray
