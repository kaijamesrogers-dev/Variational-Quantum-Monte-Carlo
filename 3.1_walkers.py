import time
start = time.time()

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,       # general font size
    'axes.titlesize': 14,  # title size
    'axes.labelsize': 14,  # x/y label size
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",   # <-- use Computer Modern
    "axes.unicode_minus": False
})

# ----------------------------------------------------------------------
# Second derivative (central difference)
# ----------------------------------------------------------------------
def w2_2nd_general(f, x, h, *f_args):
    return (f(x + h, *f_args) - 2*f(x, *f_args) + f(x - h, *f_args)) / h**2


# Initial theta for Task 3.1 (single parameter)
theta = 2.0

# ----------------------------------------------------------------------
# Wavefunction pieces and PDF
# ----------------------------------------------------------------------
def psix(x, xyz, theta):
    return np.exp(- theta * np.sqrt(x ** 2 + xyz[1] ** 2 + xyz[2] ** 2))

def psiy(x, xyz, theta):
    return np.exp(- theta * np.sqrt(xyz[0] ** 2 + x ** 2 + xyz[2] ** 2))

def psiz(x, xyz, theta):
    return np.exp(- theta * np.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + x ** 2))

def pdf_xyz(x, y, z, theta):
    r = np.sqrt(x**2 + y**2 + z**2)
    return (theta**3 / np.pi) * np.exp(-2 * theta * r)


# ----------------------------------------------------------------------
# Multi-walker 3D Metropolis–Hastings
# ----------------------------------------------------------------------
def metropolis_hastings_3d_multi(step_size, pdf, iterations, n_walkers, theta):
    """
    3D Metropolis–Hastings sampler with multiple walkers in parallel.

    Parameters
    ----------
    step_size : float
        Proposal step size in each coordinate.
    pdf : callable
        Target PDF, pdf(x, y, z, theta).
    iterations : int
        Number of Metropolis steps.
    n_walkers : int
        Number of walkers.
    theta : float
        Variational parameter.

    Returns
    -------
    samples : ndarray, shape (iterations, n_walkers, 3)
        samples[t, w, :] is the position of walker w at step t.
    acceptance_rate : float
        Overall percentage of accepted proposals (across all walkers).
    """
    samples = np.zeros((iterations, n_walkers, 3))

    # Initial points in [-1, 1]^3
    samples[0, :, :] = (np.random.rand(n_walkers, 3) - 0.5) * 2.0

    accepted_number = 0
    total_proposals = (iterations - 1) * n_walkers

    for t in range(1, iterations):
        x_current = samples[t-1, :, :]           # shape (n_walkers, 3)

        # Proposal steps for all walkers
        step_vec = step_size * (2.0 * np.random.rand(n_walkers, 3) - 1.0)
        x_prop = x_current + step_vec

        # Current and proposed pdf values
        p_current = pdf(x_current[:, 0], x_current[:, 1], x_current[:, 2], theta)
        p_prop    = pdf(x_prop[:, 0],    x_prop[:, 1],    x_prop[:, 2],    theta)

        # Avoid division by zero: if p_current <= 0, always move
        alpha = np.ones(n_walkers)
        mask_valid = p_current > 0
        alpha[mask_valid] = np.minimum(1.0, p_prop[mask_valid] / p_current[mask_valid])

        # Acceptance decisions
        u_accept = np.random.rand(n_walkers)
        accept = u_accept < alpha

        # Update samples
        samples[t, :, :] = x_current
        samples[t, accept, :] = x_prop[accept, :]

        accepted_number += np.sum(accept)

    acceptance_rate = 100.0 * accepted_number / total_proposals if total_proposals > 0 else 0.0
    return samples, acceptance_rate


# ----------------------------------------------------------------------
# Energy estimator (works with 1 walker or many)
# ----------------------------------------------------------------------
def energy(samples, theta):
    """
    Vectorised energy estimator.

    samples : array of shape (N, 3) or (iterations, n_walkers, 3)
    theta   : scalar variational parameter
    """
    samples = np.asarray(samples)
    if samples.ndim == 3:
        samples = samples.reshape(-1, 3)   # (iterations * n_walkers, 3)

    r = np.linalg.norm(samples, axis=1)    # shape (N,)
    N = samples.shape[0]

    h = 2e-4

    # Base wavefunction values ψ(r) = exp(-θ r)
    psi0 = np.exp(-theta * r)              # shape (N,)

    # Vectorised Laplacian via central differences
    lap = np.zeros_like(r)

    for dim in range(3):
        plus  = samples.copy()
        minus = samples.copy()

        plus[:, dim]  += h
        minus[:, dim] -= h

        r_plus  = np.linalg.norm(plus,  axis=1)
        r_minus = np.linalg.norm(minus, axis=1)

        psi_plus  = np.exp(-theta * r_plus)
        psi_minus = np.exp(-theta * r_minus)

        lap += (psi_plus - 2.0 * psi0 + psi_minus) / h**2

    # Avoid r = 0
    mask = r > 0
    El = np.zeros_like(r)
    El[mask] = -0.5 * lap[mask] / psi0[mask] - 1.0 / r[mask]

    return El[mask].mean()



# ----------------------------------------------------------------------
# Monte Carlo minimisation in theta with walkers
# ----------------------------------------------------------------------
def monte_carlo_minimisation(theta_step, pdf, iterations, T, theta0,
                             n_walkers=200, pos_steps=500, pos_step_size=1.1):
    """
    Metropolis-like optimisation over theta, using multi-walker Metropolis
    in position space to estimate the energy for each proposed theta.
    """
    theta = float(theta0)  # ensure scalar float
    u = np.random.rand(2 * iterations)
    n = 0

    # Initial energy at starting theta
    samples, percentage = metropolis_hastings_3d_multi(pos_step_size, pdf, pos_steps, n_walkers, theta)
    print("Initial acceptance (positions):", percentage)
    E = energy(samples, theta)

    accepted_number_mc = 0

    theta_history = [theta]
    energy_history = [E]

    for i in range(iterations):
        # propose new theta
        theta_dash = theta + theta_step * (2 * u[i] - 1)

        # sample positions for proposed theta
        samples_dash, percentage_dash = metropolis_hastings_3d_multi(
            pos_step_size, pdf, pos_steps, n_walkers, theta_dash
        )
        print(f"Step {i} position acceptance: {percentage_dash:.2f}%")
        E_dash = energy(samples_dash, theta_dash)

        delta_E = E_dash - E

        if delta_E > 0:
            alpha = np.exp(- delta_E / T)
            rnd = u[2 * i]
            if rnd < alpha:
                theta = theta_dash
                E = E_dash
                accepted_number_mc += 1
        else:
            theta = theta_dash
            E = E_dash
            accepted_number_mc += 1

        theta_history.append(theta)
        energy_history.append(E)

        n += 1
        if n == 10:
            T = T * 0.6   # cooling schedule
            n = 0

    acceptance_rate = accepted_number_mc / iterations * 100
    return theta, acceptance_rate, np.array(theta_history), np.array(energy_history)


# ----------------------------------------------------------------------
# Run the optimisation
# ----------------------------------------------------------------------
opt_theta, percentage_mc, theta_hist, E_hist = monte_carlo_minimisation(
    theta_step=0.04,
    pdf=pdf_xyz,
    iterations=200,
    T=0.05,
    theta0=theta,
    n_walkers=200,     # number of walkers
    pos_steps=1000,     # steps per walker
    pos_step_size=1.1  # position Metropolis step size
)

print("Optimized Theta:", opt_theta)
print("Theta acceptance (%):", percentage_mc)

# Final energy estimate with optimised theta
samples_optimized, percentage = metropolis_hastings_3d_multi(
    step_size=1.1,
    pdf=pdf_xyz,
    iterations=2000,
    n_walkers=200,
    theta=opt_theta
)
print("Estimated Energy Expectation Value with Optimized Theta:", energy(samples_optimized, opt_theta))
print("Position acceptance with Optimized Theta:", percentage)

# ----------------------------------------------------------------------
# Plots: theta and energy vs iteration
# ----------------------------------------------------------------------
iters = np.arange(theta_hist.shape[0])

plt.figure(figsize=(7,5))

ax1 = plt.gca()
ax2 = ax1.twinx()

# Theta curve
lns1 = ax1.plot(iters, theta_hist, '-', color='tab:blue', label=r'$\theta$')
ax1.set_xlabel("Iteration")
ax1.set_ylabel(r'$\theta$', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Energy curve
lns2 = ax2.plot(iters, E_hist, '-', color='tab:red', label='Energy')
ax2.set_ylabel("Energy", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Combine legends
lns = lns1 + lns2
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc='best')

plt.tight_layout()
plt.show()

end = time.time()
print(f"Run time: {end - start:.5f} seconds")
