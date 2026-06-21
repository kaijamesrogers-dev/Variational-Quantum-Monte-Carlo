"""
Finite-difference accuracy test for the second derivative.

Compares the 2nd-order and 4th-order central-difference approximations
to the exact second derivative of the ground-state harmonic oscillator
wavefunction, psi(x) = exp(-x^2 / 2), over a range of step sizes h.

This reproduces Fig. 1 of the project report: for large h, the error is
dominated by truncation error (scaling as h^2 or h^4). For small h, the
error is dominated by floating-point round-off, which grows as h shrinks
due to catastrophic cancellation in the numerator combined with division
by a shrinking h^2.
"""

import numpy as np
import matplotlib.pyplot as plt

# Plot styling, matched to the lab report
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})


def wf(x):
    """Ground-state harmonic oscillator wavefunction, psi(x) = exp(-x^2 / 2)."""
    return np.exp(-x**2 / 2)


def wf_2_exact(x):
    """Exact second derivative of wf(x), found analytically."""
    return (x**2 - 1) * np.exp(-x**2 / 2)


def wf_2_2nd(x, h):
    """2nd-order central-difference approximation to the second derivative."""
    return (wf(x + h) - 2 * wf(x) + wf(x - h)) / h**2


def wf_2_4th(x, h):
    """4th-order central-difference approximation to the second derivative."""
    return (-wf(x + 2 * h) + 16 * wf(x + h) - 30 * wf(x)
            + 16 * wf(x - h) - wf(x - 2 * h)) / (12 * h**2)


def first_local_minimum(h_array, errors):
    """
    Scan from large h to small h and return the first point at which the
    error stops decreasing. This locates the crossover from the
    truncation-error-dominated regime to the round-off-dominated regime,
    rather than the global minimum of a (possibly noisy) error curve.

    Parameters
    ----------
    h_array : array_like
        Step sizes, assumed ordered from small to large.
    errors : array_like
        Corresponding absolute errors.

    Returns
    -------
    idx, h_opt, err_opt : the index, step size, and error at the first
        local minimum found.
    """
    n = len(h_array)
    best_idx = n - 1
    best_err = errors[best_idx]

    for i in range(n - 2, -1, -1):
        if errors[i] < best_err:
            best_err = errors[i]
            best_idx = i
        else:
            break

    return best_idx, h_array[best_idx], errors[best_idx]


def main():
    h_array = np.logspace(-6, -0.5, 150) # 
    x0 = 0.0  # evaluation point

    errors_2nd = np.abs([wf_2_2nd(x0, h) - wf_2_exact(x0) for h in h_array])
    errors_4th = np.abs([wf_2_4th(x0, h) - wf_2_exact(x0) for h in h_array])

    idx2, h_opt_2nd, err_opt_2nd = first_local_minimum(h_array, errors_2nd)
    idx4, h_opt_4th, err_opt_4th = first_local_minimum(h_array, errors_4th)

    print(f"2nd-order: h_opt = {h_opt_2nd:.3e}, error = {err_opt_2nd:.3e}")
    print(f"4th-order: h_opt = {h_opt_4th:.3e}, error = {err_opt_4th:.3e}")

    # Reference slope lines, anchored to a point in the truncation-error
    # regime (10th-from-last h value) so they overlay the h^2 / h^4 trends.
    anchor_idx = -10
    ref_2nd = h_array**2 * errors_2nd[anchor_idx] / h_array[anchor_idx]**2
    ref_4th = h_array**4 * errors_4th[anchor_idx] / h_array[anchor_idx]**4

    plt.figure(figsize=(7, 5))
    plt.loglog(h_array, errors_2nd, label="2nd-order", linewidth=2)
    plt.loglog(h_array, errors_4th, label="4th-order", linewidth=2)
    plt.loglog(h_opt_2nd, err_opt_2nd, "o", ms=8)
    plt.loglog(h_opt_4th, err_opt_4th, "s", ms=8)
    plt.loglog(h_array, ref_2nd, "k--", label=r"$h^2$ reference")
    plt.loglog(h_array, ref_4th, "k:", label=r"$h^4$ reference")

    plt.xlabel("h")
    plt.ylabel("Error in second derivative")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()