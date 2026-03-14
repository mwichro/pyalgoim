import numpy as np
import pyalgoim
import sys
import os

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

def sample_4d(node_count, func):
    nodes = get_gauss_nodes(node_count)
    samples = np.zeros((node_count, node_count, node_count, node_count))
    for i in range(node_count):
        for j in range(node_count):
            for k in range(node_count):
                for l in range(node_count):
                    samples[i, j, k, l] = func(nodes[i], nodes[j], nodes[k], nodes[l])
    return samples

class TestPyAlgoim:
    def check_approx(self, volume, expected, tol, label="test"):
        diff = np.abs(volume - expected)
        if diff > tol:
            print(f"FAILED [{label}]: expected {expected}, got {volume} (diff {diff:.2e}, tol {tol:.2e})")
            sys.exit(1)
        else:
            print(f"PASSED [{label}]: diff {diff:.2e} (tol {tol:.2e})")

    def run_tests(self):
        print("Starting extended test suite...")
        self.test_constant_positive_2d_all()
        self.test_constant_negative_2d_all()
        self.test_straight_line_2d_y_all()
        self.test_diagonal_2d_all()
        self.test_constant_positive_3d_all()
        self.test_circle_approx_2d_all()
        self.test_sphere_approx_3d_all()
        self.test_hyperplane_4d_all()
        # Hypersphere 4D takes longer, so let's run it too
        self.test_hypersphere_approx_4d_all()
        print("All tests passed!")

    def get_inside_volume(self, quad):
        return np.sum(quad["inside"]["weights"]) if "inside" in quad else 0.0
        
    def test_constant_positive_2d_all(self):
        for nc in [3, 5]:
            gen = pyalgoim.QuadratureGenerator(2, nc)
            samples = sample_2d(nc, lambda x, y: 1.0)
            quad = gen(samples, pyalgoim.INSIDE | pyalgoim.OUTSIDE)
            self.check_approx(self.get_inside_volume(quad), 0.0, 1e-15, f"const_pos_2d_in_nc{nc}")
            self.check_approx(np.sum(quad["outside"]["weights"]), 1.0, 1e-15, f"const_pos_2d_out_nc{nc}")

    def test_constant_negative_2d_all(self):
        for nc in [3, 5]:
            gen = pyalgoim.QuadratureGenerator(2, nc)
            samples = sample_2d(nc, lambda x, y: -1.0)
            quad = gen(samples, pyalgoim.INSIDE | pyalgoim.OUTSIDE)
            self.check_approx(self.get_inside_volume(quad), 1.0, 1e-15, f"const_neg_2d_in_nc{nc}")
            vol_out = np.sum(quad["outside"]["weights"]) if "outside" in quad else 0.0
            self.check_approx(vol_out, 0.0, 1e-15, f"const_neg_2d_out_nc{nc}")

    def test_straight_line_2d_y_all(self):
        for nc in [4, 8]:
            gen = pyalgoim.QuadratureGenerator(2, nc)
            cut = 0.5
            samples = sample_2d(nc, lambda x, y: cut - y)
            quad = gen(samples, pyalgoim.INSIDE | pyalgoim.SURFACE)
            self.check_approx(self.get_inside_volume(quad), cut, 1e-15, f"straight_line_2d_in_nc{nc}")
            self.check_approx(np.sum(quad["surface"]["weights"]), 1.0, 1e-15, f"straight_line_2d_surf_nc{nc}")

    def test_diagonal_2d_all(self):
        for nc in [2, 4]:
            gen = pyalgoim.QuadratureGenerator(2, nc)
            samples = sample_2d(nc, lambda x, y: 1 - x - y)
            quad = gen(samples, pyalgoim.INSIDE)
            # Volume of triangle x+y <= 1 in [0,1]^2 is 0.5. Since level set is linear, degree 1 (nc=2) is exact.
            self.check_approx(self.get_inside_volume(quad), 0.5, 1e-15, f"diagonal_2d_nc{nc}")

    def test_constant_positive_3d_all(self):
        for nc in [2, 4]:
            gen = pyalgoim.QuadratureGenerator(3, nc)
            samples = sample_3d(nc, lambda x, y, z: 1.0)
            quad = gen(samples, pyalgoim.OUTSIDE)
            self.check_approx(np.sum(quad["outside"]["weights"]), 1.0, 1e-15, f"const_pos_3d_nc{nc}")

    def test_circle_approx_2d_all(self):
        R = 0.4
        exact_area = np.pi * R**2
        exact_perimeter = 2 * np.pi * R
        for nc in [4, 8]:
            gen = pyalgoim.QuadratureGenerator(2, nc)
            samples = sample_2d(nc, lambda x, y: ((x - 0.5)**2 + (y - 0.5)**2) - R**2)
            quad = gen(samples, pyalgoim.INSIDE | pyalgoim.SURFACE)
            volume = self.get_inside_volume(quad)
            surface_area = np.sum(quad["surface"]["weights"])
            tol_vol = 1e-5 if nc == 4 else 1e-9
            self.check_approx(volume, exact_area, tol_vol, f"circle_vol_nc{nc}")
            tol_surf = 1e-4 if nc == 4 else 1e-7
            self.check_approx(surface_area, exact_perimeter, tol_surf, f"circle_surf_nc{nc}")

    def test_sphere_approx_3d_all(self):
        R = 0.4
        exact_vol = (4/3) * np.pi * R**3
        exact_surf = 4 * np.pi * R**2
        for nc in [4]:
            gen = pyalgoim.QuadratureGenerator(3, nc)
            samples = sample_3d(nc, lambda x, y, z: ((x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2) - R**2)
            quad = gen(samples, pyalgoim.INSIDE | pyalgoim.SURFACE)
            volume = self.get_inside_volume(quad)
            surface = np.sum(quad["surface"]["weights"])
            self.check_approx(volume, exact_vol, 1e-6, f"sphere_vol_nc{nc}")
            self.check_approx(surface, exact_surf, 1e-5, f"sphere_surf_nc{nc}")

    def test_hyperplane_4d_all(self):
        for nc in [2, 4]:
            gen = pyalgoim.QuadratureGenerator(4, nc)
            # Plane x = 0.5 divides [0,1]^4 into two halves of vol 0.5
            samples = sample_4d(nc, lambda x, y, z, w: x - 0.5)
            quad = gen(samples, pyalgoim.INSIDE | pyalgoim.SURFACE)
            vol = self.get_inside_volume(quad)
            surf = np.sum(quad["surface"]["weights"])
            self.check_approx(vol, 0.5, 1e-15, f"hyperplane_4d_vol_nc{nc}")
            # Surface area of x=0.5 in [0,1]^4 is 1*1*1 = 1.0 (it's a 3D unit cube)
            self.check_approx(surf, 1.0, 1e-15, f"hyperplane_4d_surf_nc{nc}")

    def test_hypersphere_approx_4d_all(self):
        # 4D Hypersphere volume: (1/2) * pi^2 * R^4
        # 4D Hypersphere surface area (3-sphere): 2 * pi^2 * R^3
        R = 0.35
        exact_vol = 0.5 * (np.pi**2) * (R**4)
        exact_surf = 2 * (np.pi**2) * (R**3)
        for nc in [3]:
            gen = pyalgoim.QuadratureGenerator(4, nc)
            samples = sample_4d(nc, lambda x, y, z, w: ((x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2 + (w - 0.5)**2) - R**2)
            quad = gen(samples, pyalgoim.INSIDE | pyalgoim.SURFACE)
            vol = self.get_inside_volume(quad)
            surf = np.sum(quad["surface"]["weights"])
            # Lower tolerance for 4D hypersphere as it's more demanding
            self.check_approx(vol, exact_vol, 1e-5, f"hypersphere_4d_vol_nc{nc}")
            self.check_approx(surf, exact_surf, 1e-4, f"hypersphere_4d_surf_nc{nc}")

if __name__ == "__main__":
    if "ALGOIM_PY_BACKEND" not in os.environ:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        backend = os.path.join(os.path.dirname(test_dir), "build", "pyalgoim_backend.so")
        if os.path.exists(backend):
            os.environ["ALGOIM_PY_BACKEND"] = backend
    
    TestPyAlgoim().run_tests()
