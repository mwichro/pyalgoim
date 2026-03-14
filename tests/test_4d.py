import numpy as np
import pyalgoim
import sys

def get_gauss_nodes(n):
    nodes, _ = np.polynomial.legendre.leggauss(n)
    return (nodes + 1.0) / 2.0

def sample_4d(node_count, func):
    nodes = get_gauss_nodes(node_count)
    samples = np.zeros((node_count, node_count, node_count, node_count))
    for i in range(node_count):
        for j in range(node_count):
            for k in range(node_count):
                for l in range(node_count):
                    samples[i, j, k, l] = func(nodes[i], nodes[j], nodes[k], nodes[l])
    return samples

def test_4d_hyperplane():
    print("Testing 4D hyperplane...")
    node_count = 4
    # Plane x + y + z + w - 1.5 = 0
    # Inside: x + y + z + w < 1.5
    gen = pyalgoim.QuadratureGenerator(4, node_count)
    samples = sample_4d(node_count, lambda x, y, z, w: x + y + z + w - 1.5)
    
    quad = gen(samples, pyalgoim.INSIDE | pyalgoim.OUTSIDE | pyalgoim.SURFACE)
    
    vol_in = np.sum(quad["inside"]["weights"])
    vol_out = np.sum(quad["outside"]["weights"])
    
    print(f"4D Volume Inside: {vol_in}")
    print(f"4D Volume Outside: {vol_out}")
    print(f"Total Volume: {vol_in + vol_out}")
    
    assert np.isclose(vol_in + vol_out, 1.0, atol=1e-12)
    assert vol_in > 0 and vol_out > 0
    
    if "surface" in quad:
        surf_weights = quad["surface"]["weights"]
        print(f"4D Surface point count: {len(surf_weights)}")
        assert len(surf_weights) > 0

if __name__ == "__main__":
    try:
        test_4d_hyperplane()
        print("4D test passed!")
    except Exception as e:
        print(f"4D test failed: {e}")
        sys.exit(1)
