"""Test for hilbert_bytes."""

import hilbert
import numpy as np
import pytest

import hilbert_bytes


def test_single_dim() -> None:
    """Test hilbert index on single dimension.

    This should be a noop.
    """
    nums = np.arange(0, 1 << 16, dtype="u2")
    byte_nums = nums[..., None].astype(">u2").view("u1")

    expected_points = hilbert.decode(nums, 1, 16)
    byte_points = hilbert_bytes.decode(byte_nums, 1)
    actual_points = byte_points.view(">u2").astype("u8")[..., 0]
    assert np.all(expected_points == actual_points)

    expected_nums = hilbert.encode(expected_points, 1, 16)
    assert np.all(nums == expected_nums)
    actual_nums = hilbert_bytes.encode(byte_points)
    assert np.all(byte_nums == actual_nums)


def test_two_dims() -> None:
    """Test hilbert bytes on two dimensions."""
    nums = np.arange(0, 1 << 16, dtype="u2")
    byte_nums = nums[..., None].astype(">u2").view("u1")

    expected_points = hilbert.decode(nums, 2, 8)
    byte_points = hilbert_bytes.decode(byte_nums, 2)
    assert np.all(expected_points == byte_points[..., 0])

    expected_nums = hilbert.encode(expected_points, 2, 8)
    assert np.all(nums == expected_nums)
    actual_nums = hilbert_bytes.encode(byte_points)
    assert np.all(byte_nums == actual_nums)


def test_three_dims() -> None:
    """Test with three dimensions."""
    nums = np.arange(0, 1 << 24, 53, dtype="u4")
    byte_nums = nums[..., None].astype(">u4").view("u1")[:, 1:]

    expected_points = hilbert.decode(nums, 3, 8)
    byte_points = hilbert_bytes.decode(byte_nums, 3)
    assert np.all(expected_points == byte_points[..., 0])

    expected_nums = hilbert.encode(expected_points, 3, 8)
    assert np.all(nums == expected_nums)
    actual_nums = hilbert_bytes.encode(byte_points)
    assert np.all(byte_nums == actual_nums)


def test_exception() -> None:
    """Test exception when bytes don't align."""
    nums = np.arange(0, 1 << 8, dtype="u1")[:, None]
    with pytest.raises(ValueError):
        hilbert_bytes.decode(nums, 2)
