import numpy as np
import matplotlib.pyplot as plt

#Match text to lab report
plt.rcParams.update({
    'font.size': 14, 
    'axes.titlesize': 14,  
    'axes.labelsize': 14, 
    'ytick.labelsize': 14,
    'legend.fontsize': 14})

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False})

# harmonic oscillator ground state wavefunction
def wf(x):
    return np.exp(-x**2 / 2)

# Analytic second derivative of exp(-x^2/2)
def wf_2_exact(x):
    return (x**2 - 1) * np.exp(-x**2 / 2)

# 2nd order central difference
def wf_2_2nd(x, h):
    return (wf(x + h) - 2 * wf(x) + wf(x - h)) / h**2

# 4th order central difference
def wf_2_4th(x, h):
    return (-wf(x + 2*h) + 16*wf(x + h) - 30*wf(x) +
            16*wf(x - h) - wf(x - 2*h)) / (12 * h**2)

h_array = np.logspace(-6, -0.5, 150)
x0 = 0.998

errors_2nd = []
errors_4th = []

for h in h_array:
    errors_2nd.append(abs(wf_2_2nd(x0, h) - wf_2_exact(x0)))
    errors_4th.append(abs(wf_2_4th(x0, h) - wf_2_exact(x0)))

errors_2nd = np.array(errors_2nd)
errors_4th = np.array(errors_4th)

# best h before round-off dominates finder
def best_before_roundoff(h_array, errors):
    """
    Go from large h to small h and stop when error stops improving.
    """
    n = len(h_array)
    best_idx = n - 1
    best_err = errors[best_idx]

    # move towards small h
    for i in range(n - 2, -1, -1):
        if errors[i] < best_err:
            best_err = errors[i]
            best_idx = i
        else:
            break

    return best_idx, h_array[best_idx], errors[best_idx]

idx2, h_opt_2nd, err_opt_2nd = best_before_roundoff(h_array, errors_2nd)
idx4, h_opt_4th, err_opt_4th = best_before_roundoff(h_array, errors_4th)

print("2nd-order pre roundoff: h_opt=", h_opt_2nd, " error=", err_opt_2nd)
print("4th-order pre roundoff: h_opt=", h_opt_4th, " error=", err_opt_4th)

# plot
plt.figure(figsize=(7, 5))
plt.loglog(h_array, errors_2nd, label="2nd-order", linewidth=2)
plt.loglog(h_array, errors_4th, label="4th-order", linewidth=2)

plt.loglog(h_opt_2nd, err_opt_2nd, 'o', ms=8)
plt.loglog(h_opt_4th, err_opt_4th, 's', ms=8)

plt.loglog(h_array, h_array**2 * errors_2nd[-10] / h_array[-10]**2, 'k--')
plt.loglog(h_array, h_array**4 * errors_4th[-10] / h_array[-10]**4, 'k:')

plt.xlabel("h")
plt.ylabel("Error in second derivative")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.show()