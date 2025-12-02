import numpy as np

def phi_3d(r, a=0.5):
    """Test function φ(r) = exp(-a * |r|^2), r is a 3-component array."""
    r2 = np.dot(r, r)
    return np.exp(-a * r2)

def laplacian_phi_exact(r, a=0.5):
    """Exact Laplacian ∇^2 φ for φ(r) = exp(-a r^2) in 3D."""
    r2 = np.dot(r, r)
    return (4.0 * a**2 * r2 - 6.0 * a) * np.exp(-a * r2)

def laplacian_3d_fd(f, r, h, a=0.5):
    """
    Central-difference Laplacian of f(r) in 3D at point r.

    f should be a function f(r, a) where r is length-3 array.
    """
    base = f(r, a)
    lap = 0.0

    for dim in range(3):
        e = np.zeros(3)
        e[dim] = 1.0

        f_plus  = f(r + h * e, a)
        f_minus = f(r - h * e, a)

        lap += (f_plus - 2.0 * base + f_minus)

    return lap / (h**2)

def test_3d_laplacian():
    # Test points in space
    test_points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.0]),
        np.array([1.0, 0.5, 0.2])
    ]
    hs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 0.5e-3, 1e-4, 1e-5]
    a = 0.5

    print("\nTesting 3D Laplacian finite difference:")
    for h in hs:
        errors = []
        for r in test_points:
            num = laplacian_3d_fd(phi_3d, r, h, a)
            exact = laplacian_phi_exact(r, a)
            errors.append(abs(num - exact))
        max_err = max(errors)
        rms_err = np.sqrt(np.mean(np.array(errors)**2))
        print(f"h = {h:8.1e}  max error = {max_err:.3e},  RMS error = {rms_err:.3e}")

if __name__ == "__main__":
    test_3d_laplacian()
