import numpy as np
import pyalgoim
import sys

def get_gauss_nodes(n):
    nodes, _ = np.polynomial.legendre.leggauss(n)
    return (nodes + 1.0) / 2.0

def sample_2d(node_count, func):
    nodes = get_gauss_nodes(node_count)
    samples = np.zeros((node_count, node_count))
    for i in range(node_count):
        for j in range(node_count):
            samples[i, j] = func(nodes[i], nodes[j])
    return samples

def sample_3d(node_count, func):
    nodes = get_gauss_nodes(node_count)
    samples = np.zeros((node_count, node_count, node_count))
    for i in range(node_count):
        for j in range(node_count):
            for k in range(node_count):
                samples[i, j, k] = func(nodes[i], nodes[j], nodes[k])
    return samples

class TestPyAlgoim:
    def check_approx(self, volume, expected, tol):
        if not np.abs(volume - expected) < tol:
            print(f"Error: expected {expected}, got {volume}")
            sys.exit(1)

    def run_tests(self):
        self.test_constant_positive_2d_all()
        self.test_constant_negative_2d_all()
        self.test_straight_line_2d_y_all()
        self.test_diagonal_2d_all()
        self.test_constant_positive_3d_all()
        self.test_circle_approx_2d_all()
        self.test_sphere_approx_3d_all()
        print("All tests passed!")

    def get_inside_volume(self, quad):
        return np.sum(quad["inside"]["weights"]) if "inside" in quad else 0.0
        
    def test_constant_positive_2d_all(self):
        for nc in [3, 5, 7]:
            gen = pyalgoim.QuadratureGenerator(2, nc)
            samples = sample_2d(nc, lambda x, y: 1.0)
            quad = gen(samples, pyalgoim.INSIDE)
            assert np.isclose(self.get_inside_volume(quad), 0.0)
            quad = gen(samples, pyalgoim.OUTSIDE)
            assert np.isclose(np.sum(quad["outside"]["weights"]), 1.0)

    def test_constant_negative_2d_all(self):
        for nc in [3, 5, 7]:
            gen = pyalgoim.QuadratureGenerator(2, nc)
            samples = sample_2d(nc, lambda x, y: -1.0)
            quad = gen(samples, pyalgoim.INSIDE)
            assert np.isclose(self.get_inside_volume(quad), 1.0)
            quad = gen(samples, pyalgoim.OUTSIDE)
            assert np.isclose(np.sum(quad["outside"]["weights"]) if "outside" in quad else 0.0, 0.0)

    def test_straight_line_2d_y_all(self):
        for nc in [3, 5]:
            gen = pyalgoim.QuadratureGenerator(2, nc)
            cut = 0.5
            samples = sample_2d(nc, lambda x, y: cut - y)
            quad = gen(samples, pyalgoim.INSIDE | pyalgoim.SURFACE)
            assert np.isclose(self.get_inside_volume(quad), cut)
            # Length of cutoff line is exactly 1.0 across a [0,1]^2 cell at y=0.5
            assert np.isclose(np.sum(quad["surface"]["weights"]), 1.0)

    def test_diagonal_2d_all(self):
        for nc in [3, 5]:
            gen = pyalgoim.QuadratureGenerator(2, nc)
            samples = sample_2d(nc, lambda x, y: 1 - x - y)
            quad = gen(samples, pyalgoim.INSIDE)
            # Volume of triangle x+y <= 1 in [0,1]^2 
            assert np.isclose(self.get_inside_volume(quad), 0.5)

    def test_constant_positive_3d_all(self):
        for nc in [2, 3, 5]:
            gen = pyalgoim.QuadratureGenerator(3, nc)
            samples = sample_3d(nc, lambda x, y, z: 1.0)
            quad = gen(samples, pyalgoim.OUTSIDE)
            assert np.isclose(np.sum(quad["outside"]["weights"]), 1.0)

    def test_circle_approx_2d_all(self):
        R = 0.4
        exact_area = np.pi * R**2
        exact_perimeter = 2 * np.pi * R
        for nc in [3, 5, 7, 9]:
            gen = pyalgoim.QuadratureGenerator(2, nc)
            # Level set function phi = R^2 - ((x-0.5)^2 + (y-0.5)^2). Inside is phi < 0, outside is phi > 0.
            # We want inside the circle, so phi should be > 0 inside. Wait, Algoim convention: phi_i < 0 is "inside".
            # So phi = ((x-0.5)^2 + (y-0.5)^2) - R^2 means inside is phi < 0.
            samples = sample_2d(nc, lambda x, y: ((x - 0.5)**2 + (y - 0.5)**2) - R**2)
            quad = gen(samples, pyalgoim.INSIDE | pyalgoim.SURFACE)
            volume = self.get_inside_volume(quad)
            surface_area = np.sum(quad["surface"]["weights"])
            tol = 0.1 / nc
            self.check_approx(volume, exact_area, tol)
            # Surface measure is a bit more sensitive, but let's check it converges too.
            self.check_approx(surface_area, exact_perimeter, tol * 2.0)

    def test_sphere_approx_3d_all(self):
        R = 0.4
        exact_vol = (4/3) * np.pi * R**3
        for nc in [3, 5]:
            gen = pyalgoim.QuadratureGenerator(3, nc)
            samples = sample_3d(nc, lambda x, y, z: ((x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2) - R**2)
            quad = gen(samples, pyalgoim.INSIDE)
            volume = self.get_inside_volume(quad)
            tol = 0.2 / nc
            self.check_approx(volume, exact_vol, tol)

if __name__ == "__main__":
    TestPyAlgoim().run_tests()
