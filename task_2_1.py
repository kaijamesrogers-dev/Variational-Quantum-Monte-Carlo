import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,       # general font size
    'axes.titlesize': 14,  # title size
    'axes.labelsize': 14,  # x/y label size
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14})

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",   # <-- use Computer Modern
    "axes.unicode_minus": False})

# Exact test wavefunction: harmonic oscillator ground state
def w(x):
    return np.exp(-x**2 / 2)

def w2_exact(x):
    # Analytic second derivative of exp(-x^2/2)
    return (x**2 - 1) * np.exp(-x**2 / 2)

# 3-point (2nd order) central difference
def w2_2nd(x, h):
    return (w(x + h) - 2 * w(x) + w(x - h)) / h**2

# 5-point (4th order) central difference
def w2_4th(x, h):
    return (-w(x + 2*h) + 16*w(x + h) - 30*w(x) +
            16*w(x - h) - w(x - 2*h)) / (12 * h**2)

# Range of h values to test
hs = np.logspace(-6, -0.5, 150)
x0 = 0.998   # point at which we test the derivative

errors_2nd = []
errors_4th = []

for h in hs:
    errors_2nd.append(abs(w2_2nd(x0, h) - w2_exact(x0)))
    errors_4th.append(abs(w2_4th(x0, h) - w2_exact(x0)))

errors_2nd = np.array(errors_2nd)
errors_4th = np.array(errors_4th)

# -------- helper: best h before round-off dominates --------
def best_before_roundoff(hs, errors):
    """
    Scan from large h to small h.
    As long as the error keeps decreasing when we reduce h, we update 'best'.
    The first time the error stops decreasing, we assume round-off is starting
    to dominate and return the last best point.
    """
    n = len(hs)
    # start from the largest h (right end of the array)
    best_idx = n - 1
    best_err = errors[best_idx]

    # walk towards smaller h
    for i in range(n - 2, -1, -1):
        if errors[i] < best_err:
            # still improving: update best
            best_err = errors[i]
            best_idx = i
        else:
            # first time it stops improving -> we've hit the "turning point"
            break

    return best_idx, hs[best_idx], errors[best_idx]

idx2, h_opt_2nd, err_opt_2nd = best_before_roundoff(hs, errors_2nd)
idx4, h_opt_4th, err_opt_4th = best_before_roundoff(hs, errors_4th)

print("2nd-order (pre-roundoff): h_opt ≈", h_opt_2nd, " error ≈", err_opt_2nd)
print("4th-order (pre-roundoff): h_opt ≈", h_opt_4th, " error ≈", err_opt_4th)

# For comparison, the true global minima (usually in the noisy round-off zone)
idx2_glob = np.argmin(errors_2nd)
idx4_glob = np.argmin(errors_4th)
print("2nd-order (global min):   h ≈", hs[idx2_glob], " error ≈", errors_2nd[idx2_glob])
print("4th-order (global min):   h ≈", hs[idx4_glob], " error ≈", errors_4th[idx4_glob])

# ------------------------------- Plot -------------------------------------
plt.figure(figsize=(7, 5))
plt.loglog(hs, errors_2nd, label="2nd-order", linewidth=2)
plt.loglog(hs, errors_4th, label="4th-order", linewidth=2)

# Mark "best before roundoff" points
plt.loglog(h_opt_2nd, err_opt_2nd, 'o', ms=8)
plt.loglog(h_opt_4th, err_opt_4th, 's', ms=8)

# Plot reference slopes h^2 and h^4
plt.loglog(hs,
           hs**2 * errors_2nd[-10] / hs[-10]**2,
           'k--')
plt.loglog(hs,
           hs**4 * errors_4th[-10] / hs[-10]**4,
           'k:')

plt.xlabel("h")
plt.ylabel("Error in second derivative")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------
# General 3-point (2nd order) central difference function
def w2_2nd_general(f, x, h, *f_args):
    return (f(x + h, *f_args) - 2*f(x, *f_args) + f(x - h, *f_args)) / h**2
    return (f(x + h, *f_args) - 2*f(x, *f_args) + f(x - h, *f_args)) / h**2



