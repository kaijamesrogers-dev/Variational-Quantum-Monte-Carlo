import numpy as np
import matplotlib.pyplot as plt

# Exact test wavefunction: harmonic oscillator ground state
def w(x):
    return np.exp(-x**2/2)

def w2_exact(x):
    # analytic second derivative of exp(-x^2/2)
    return (x**2 - 1) * np.exp(-x**2/2)

# 3-point (2nd order) central difference
def w2_2nd(x, h):
    return (w(x+h) - 2*w(x) + w(x-h)) / h**2

# 5-point (4th order) central difference
def w2_4th(x, h):
    return (-w(x+2*h) + 16*w(x+h) - 30*w(x) + 16*w(x-h) - w(x-2*h)) / (12*h**2)

# Range of h values to test
hs = np.logspace(-6, -0.5, 150)
x0 = 0.998   # choose any point you like

errors_2nd = []
errors_4th = []

for h in hs:
    errors_2nd.append(abs(w2_2nd(x0,h) - w2_exact(x0)))
    errors_4th.append(abs(w2_4th(x0,h) - w2_exact(x0)))

# Plot
plt.figure(figsize=(7,5))
plt.loglog(hs, errors_2nd, label="2nd-order (3-point)", linewidth=2)
plt.loglog(hs, errors_4th, label="4th-order (5-point)", linewidth=2)

# Plot reference slopes h^2 and h^4
plt.loglog(hs, hs**2 * errors_2nd[10]/hs[10]**2, 'k--', label='slope = 2')
plt.loglog(hs, hs**4 * errors_4th[10]/hs[10]**4, 'k:', label='slope = 4')

plt.xlabel("h")
plt.ylabel("Error in second derivative")
plt.title("Finite Difference Error Scaling")
plt.legend()
plt.grid(True, which="both")
plt.show()

#----------------------------------------------------------------------------
# general 3-point (2nd order) central difference function
def w2_2nd_general(f, x, h, *f_args):
    return (f(x + h, *f_args) - 2*f(x, *f_args) + f(x - h, *f_args)) / h**2



