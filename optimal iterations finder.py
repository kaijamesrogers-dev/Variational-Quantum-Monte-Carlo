import numpy as np
import matplotlib.pyplot as plt
from task_2_2 import metropolis_hastings_multi, pdf0

# --- your existing sampling call ---
samples_multi, fraction = metropolis_hastings_multi(
    step_size=2,
    pdf=pdf0,           # your target PDF
    iterations=5000,
    n_walkers=100
)

# --- define local energy for 1D HO ground state ---
def local_energy_ho(x):
    # For psi(x) ∝ exp(-x^2/2), EL(x) = 0.5 * (x^2 + 1)
    return 0.5 * (x**2 + 1.0)

# Compute local energies for all samples
E_samples = local_energy_ho(samples_multi)

# Running mean of the energy as a function of sample index
running_E = np.cumsum(E_samples) / np.arange(1, len(E_samples) + 1)

# Optional: convert x-axis to "effective iterations" (sweeps per walker)
iterations = 1 + (np.arange(len(running_E)) // 10000)  # if n_walkers = 10000

# --- Plot running energy vs iterations ---
plt.figure(figsize=(7, 5))
plt.plot(iterations, running_E, label="Running energy estimate")
plt.axhline(0.5, linestyle='--', label="Exact E0 = 0.5")  # exact ground-state energy

plt.xlabel("Effective iterations (sweeps per walker)")
plt.ylabel("Energy estimate")
plt.title("Convergence of energy with Metropolis iterations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
