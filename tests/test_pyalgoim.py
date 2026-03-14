from __future__ import annotations

import numpy as np

import pyalgoim


TOL = 5.0e-5


def assert_close(actual: float, expected: float, message: str) -> None:
    if abs(actual - expected) > TOL:
        raise AssertionError(f"{message}: expected {expected}, got {actual}")


def sum_weights(entry: dict[str, np.ndarray]) -> float:
    return float(entry["weights"].sum())


def sample_2d(node_count: int, func) -> np.ndarray:
    cell = np.empty((node_count, node_count), dtype=float)
    gauss_nodes = [0.1127016653792583, 0.5, 0.8872983346207417] if node_count == 3 else None
    if gauss_nodes is None:
        raise AssertionError("test helper only implements node_count=3")
    for i, x in enumerate(gauss_nodes):
        for j, y in enumerate(gauss_nodes):
            cell[i, j] = func(x, y)
    return cell


def sample_3d(node_count: int, func) -> np.ndarray:
    cell = np.empty((node_count, node_count, node_count), dtype=float)
    gauss_nodes = [0.1127016653792583, 0.5, 0.8872983346207417] if node_count == 3 else None
    if gauss_nodes is None:
        raise AssertionError("test helper only implements node_count=3")
    for i, x in enumerate(gauss_nodes):
        for j, y in enumerate(gauss_nodes):
            for k, z in enumerate(gauss_nodes):
                cell[i, j, k] = func(x, y, z)
    return cell


def test_constant_values() -> None:
    node_count = 3
    positive = np.ones((node_count, node_count), dtype=float)
    negative = -np.ones((node_count, node_count), dtype=float)

    positive_result = pyalgoim.generate_cell_quadratures(positive, pyalgoim.INSIDE | pyalgoim.OUTSIDE | pyalgoim.SURFACE)
    negative_result = pyalgoim.generate_cell_quadratures(negative, pyalgoim.INSIDE | pyalgoim.OUTSIDE | pyalgoim.SURFACE)

    assert_close(sum_weights(positive_result["inside"]), 0.0, "positive constant field should have zero inside measure")
    assert_close(sum_weights(positive_result["outside"]), 1.0, "positive constant field should have full outside measure")
    assert positive_result["surface"]["weights"].size == 0
    assert_close(sum_weights(negative_result["inside"]), 1.0, "negative constant field should have full inside measure")
    assert_close(sum_weights(negative_result["outside"]), 0.0, "negative constant field should have zero outside measure")
    assert negative_result["surface"]["weights"].size == 0


def test_straight_line_2d() -> None:
    cell = sample_2d(3, lambda x, y: x - 0.5)
    result = pyalgoim.generate_cell_quadratures(cell, pyalgoim.INSIDE | pyalgoim.OUTSIDE | pyalgoim.SURFACE)

    assert_close(sum_weights(result["inside"]), 0.5, "2D line split should produce half-cell inside measure")
    assert_close(sum_weights(result["outside"]), 0.5, "2D line split should produce half-cell outside measure")
    assert_close(sum_weights(result["surface"]), 1.0, "2D line split should produce unit interface length")
    assert np.allclose(result["surface"]["points"][:, 0], 0.5, atol=TOL)


def test_constant_values_3d() -> None:
    positive = np.ones((3, 3, 3), dtype=float)
    negative = -np.ones((3, 3, 3), dtype=float)

    positive_result = pyalgoim.generate_cell_quadratures(positive, pyalgoim.INSIDE | pyalgoim.OUTSIDE | pyalgoim.SURFACE, spatial_dimension=3)
    negative_result = pyalgoim.generate_cell_quadratures(negative, pyalgoim.INSIDE | pyalgoim.OUTSIDE | pyalgoim.SURFACE, spatial_dimension=3)

    assert positive_result["spatial_dimension"] == 3
    assert_close(sum_weights(positive_result["inside"]), 0.0, "positive constant 3D field should have zero inside measure")
    assert_close(sum_weights(positive_result["outside"]), 1.0, "positive constant 3D field should have full outside measure")
    assert positive_result["surface"]["weights"].size == 0
    assert_close(sum_weights(negative_result["inside"]), 1.0, "negative constant 3D field should have full inside measure")
    assert_close(sum_weights(negative_result["outside"]), 0.0, "negative constant 3D field should have zero outside measure")
    assert negative_result["surface"]["weights"].size == 0


if __name__ == "__main__":
    test_constant_values()
    test_straight_line_2d()
    test_constant_values_3d()
    print("pyalgoim tests passed")