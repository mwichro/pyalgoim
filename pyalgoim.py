from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np


INSIDE = 1 << 0
OUTSIDE = 1 << 1
SURFACE = 1 << 2


class _PackedQuadrature(ctypes.Structure):
    _fields_ = [
        ("points", ctypes.POINTER(ctypes.c_double)),
        ("weights", ctypes.POINTER(ctypes.c_double)),
        ("offsets", ctypes.POINTER(ctypes.c_uint64)),
        ("point_dimension", ctypes.c_int),
        ("point_count", ctypes.c_uint64),
        ("offset_count", ctypes.c_uint64),
    ]


class _Handle(ctypes.Structure):
    pass


def _find_backend() -> Path:
    env_path = os.environ.get("ALGOIM_PY_BACKEND")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    root = Path(__file__).resolve().parent
    candidates.extend(
        [
            root / "build" / "pyalgoim_backend.so",
            root / "build" / "algoim_batch_backend.so",
            root / "pyalgoim_backend.so",
            root / "algoim_batch_backend.so",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate pyalgoim_backend.so. Build the project with CMake first or set ALGOIM_PY_BACKEND."
    )


_lib = ctypes.CDLL(str(_find_backend()))

_lib.algoim_generator_create.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint64
]
_lib.algoim_generator_create.restype = ctypes.POINTER(_Handle)

_lib.algoim_generator_free.argtypes = [ctypes.POINTER(_Handle)]
_lib.algoim_generator_free.restype = None

_lib.algoim_generate_batch_quadrature.argtypes = [
    ctypes.POINTER(_Handle),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_char_p,
    ctypes.c_uint64,
]
_lib.algoim_generate_batch_quadrature.restype = ctypes.POINTER(_Handle)

_lib.algoim_batch_quadrature_free.argtypes = [ctypes.POINTER(_Handle)]
_lib.algoim_batch_quadrature_free.restype = None

for name in ["inside", "outside", "surface"]:
    getattr(_lib, f"algoim_batch_quadrature_has_{name}").argtypes = [ctypes.POINTER(_Handle)]
    getattr(_lib, f"algoim_batch_quadrature_has_{name}").restype = ctypes.c_int
    getattr(_lib, f"algoim_batch_quadrature_get_{name}").argtypes = [ctypes.POINTER(_Handle), ctypes.POINTER(_PackedQuadrature)]
    getattr(_lib, f"algoim_batch_quadrature_get_{name}").restype = ctypes.c_int


def _as_numpy(packed: _PackedQuadrature) -> dict[str, np.ndarray]:
    point_count = int(packed.point_count)
    point_dimension = int(packed.point_dimension)
    offset_count = int(packed.offset_count)

    if point_count:
        points = np.ctypeslib.as_array(packed.points, shape=(point_count * point_dimension,)).copy().reshape(point_count, point_dimension)
        weights = np.ctypeslib.as_array(packed.weights, shape=(point_count,)).copy()
    else:
        points = np.empty((0, point_dimension), dtype=np.float64)
        weights = np.empty((0,), dtype=np.float64)

    if offset_count:
        offsets = np.ctypeslib.as_array(packed.offsets, shape=(offset_count,)).copy()
    else:
        offsets = np.empty((0,), dtype=np.uint64)

    return {"points": points, "weights": weights, "offsets": offsets}



class QuadratureGenerator:
    def __init__(self, spatial_dimension: int, node_count: int, basis: str = "monomial", line_rule: str = "gauss-legendre", line_points: int | None = None):
        self.spatial_dimension = spatial_dimension
        self.node_count = node_count
        basis_type = 1 if basis.lower() == "monomial" else 0
        error_buffer = ctypes.create_string_buffer(1024)
        
        self._handle = _lib.algoim_generator_create(
            int(spatial_dimension),
            int(node_count),
            int(basis_type),
            line_rule.encode("utf-8"),
            -1 if line_points is None else int(line_points),
            error_buffer,
            len(error_buffer)
        )
        if not self._handle:
            raise RuntimeError(error_buffer.value.decode("utf-8"))

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            _lib.algoim_generator_free(self._handle)

    def __call__(self, cells: Any, flags: int) -> dict[str, Any]:
        cells_array = np.ascontiguousarray(cells, dtype=np.float64)
        
        if cells_array.ndim == 2:
            batch_size = 1
        elif cells_array.ndim == 3 and self.spatial_dimension == 3:
            batch_size = 1
        elif cells_array.ndim == 3 and self.spatial_dimension == 2:
            batch_size = int(cells_array.shape[0])
        elif cells_array.ndim == 4 and self.spatial_dimension == 3:
            batch_size = int(cells_array.shape[0])
        else:
            raise ValueError("Invalid cells shape for the generator's spatial dimension")

        error_buffer = ctypes.create_string_buffer(1024)
        handle = _lib.algoim_generate_batch_quadrature(
            self._handle,
            cells_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            batch_size,
            flags,
            error_buffer,
            len(error_buffer),
        )
        if not handle:
            raise RuntimeError(error_buffer.value.decode("utf-8"))
            
        try:
            result: dict[str, Any] = {}
            for name in ["inside", "outside", "surface"]:
                if getattr(_lib, f"algoim_batch_quadrature_has_{name}")(handle):
                    packed = _PackedQuadrature()
                    getattr(_lib, f"algoim_batch_quadrature_get_{name}")(handle, ctypes.byref(packed))
                    result[name] = _as_numpy(packed)
            return result
        finally:
            _lib.algoim_batch_quadrature_free(handle)

def generate_cell_quadratures(
    cells: Any,
    flags: int,
    line_rule: str = "gauss-legendre",
    line_points: int | None = None,
    spatial_dimension: int | None = None,
) -> dict[str, Any]:
    cells_array = np.ascontiguousarray(cells, dtype=np.float64)
    if cells_array.ndim not in (2, 3, 4):
        raise ValueError("cells must have shape (batch, n, n), (n, n), (batch, n, n, n), or (n, n, n)")

    if cells_array.ndim == 2:
        if spatial_dimension not in (None, 2):
            raise ValueError("2D cell arrays require spatial_dimension=2 or None")
        if cells_array.shape[0] != cells_array.shape[1]:
            raise ValueError("single 2D cell input must have shape (n, n)")
        batch_size = 1
        spatial_dimension = 2
        node_count = int(cells_array.shape[0])
    elif cells_array.ndim == 3 and spatial_dimension == 3:
        if cells_array.shape[0] != cells_array.shape[1] or cells_array.shape[1] != cells_array.shape[2]:
            raise ValueError("single 3D cell input must have shape (n, n, n)")
        batch_size = 1
        spatial_dimension = 3
        node_count = int(cells_array.shape[0])
    elif cells_array.ndim == 3:
        if spatial_dimension not in (None, 2):
            raise ValueError("3D arrays are interpreted as batched 2D cells unless spatial_dimension=3 is set")
        if cells_array.shape[1] != cells_array.shape[2]:
            raise ValueError("batched 2D input must have shape (batch, n, n) with equal node counts per axis")
        batch_size = int(cells_array.shape[0])
        spatial_dimension = 2
        node_count = int(cells_array.shape[1])
    else:
        if cells_array.shape[1] != cells_array.shape[2] or cells_array.shape[2] != cells_array.shape[3]:
            raise ValueError("batched 3D input must have shape (batch, n, n, n) with equal node counts per axis")
        batch_size = int(cells_array.shape[0])
        spatial_dimension = 3
        node_count = int(cells_array.shape[1])

    gen = QuadratureGenerator(spatial_dimension, node_count, "monomial", line_rule, line_points)
    return gen(cells_array, flags)
