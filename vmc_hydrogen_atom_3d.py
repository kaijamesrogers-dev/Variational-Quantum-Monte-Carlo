"""
VMC for the hydrogen atom in 3D.
 
The trial wavefunction is now psi(r; theta) = exp(-theta * r) where theta is a
free parameter that has to be found.
 
The ground-state theta is the value of theta that minimises the expectation
energy.
 
Simulated annealing is used instead to search over theta: at each
step a new theta is proposed, its energy is estimated via a
fresh Monte Carlo run, and the move is accepted if the energy
decreases, or accepted anyway with probability exp(-delta_E / T) if it
increases. This lets the search escape local minima.
"""

import time

import numpy as np
import matplotlib.pyplot as plt

# Match text to lab report
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


def w2_2nd_general(f, x, h, *f_args):
    """General 2nd-order central-difference second derivative of f at x."""
    return (f(x + h, *f_args) - 2 * f(x, *f_args) + f(x - h, *f_args)) / h**2


def pdf_xyz(x, y, z, theta):
    """|psi(r; theta)|^2 for the trial wavefunction psi = exp(-theta * r)."""
    r = np.sqrt(x**2 + y**2 + z**2)
    return (theta**3 / np.pi) * np.exp(-2 * theta * r)


def metropolis_3d_multi(step_size, pdf, iterations, n_walkers, theta):
    """
    3D Metropolis sampler with multiple walkers evolved in parallel.

    Parameters
    ----------
    step_size : float
        Half-width of the uniform proposal distribution (per coordinate).
    pdf : callable
        Target probability density, called as pdf(x, y, z, theta).
    iterations : int
        Number of steps in each chain.
    n_walkers : int
        Number of independent walkers run in parallel.
    theta : float
        Variational parameter passed through to pdf.

    Returns
    -------
    samples : ndarray, shape (iterations, n_walkers, 3)
        Sampled 3D positions for every walker at every iteration.
    acceptance_rate : float
        Percentage of proposed moves that were accepted.
    """
    samples = np.zeros((iterations, n_walkers, 3))

    # initial points in [-1, 1]^3
    samples[0, :, :] = (np.random.rand(n_walkers, 3) - 0.5) * 2.0

    accepted_number = 0
    total_proposals = (iterations - 1) * n_walkers

    for t in range(1, iterations):
        x_current = samples[t - 1, :, :]

        # proposal steps for all walkers
        step_vec = step_size * (2.0 * np.random.rand(n_walkers, 3) - 1.0)
        x_prop = x_current + step_vec

        # current and proposed pdf values
        p_current = pdf(x_current[:, 0], x_current[:, 1], x_current[:, 2], theta)
        p_prop = pdf(x_prop[:, 0], x_prop[:, 1], x_prop[:, 2], theta)

        # avoid division by zero
        alpha = np.ones(n_walkers)
        mask_valid = p_current > 0
        alpha[mask_valid] = np.minimum(1.0, p_prop[mask_valid] / p_current[mask_valid])

        # acceptance decisions
        u_accept = np.random.rand(n_walkers)
        accept = u_accept < alpha

        # update samples
        samples[t, :, :] = x_current
        samples[t, accept, :] = x_prop[accept, :]

        accepted_number += np.sum(accept)

    acceptance_rate = 100.0 * accepted_number / total_proposals if total_proposals > 0 else 0.0
    return samples, acceptance_rate


def energy(samples, theta):
    """
    Monte Carlo estimate of <H> for the hydrogen atom given sampled
    electron positions, using the local energy

        E_l(r) = -1/2 * (laplacian psi) / psi - 1/r

    The Laplacian is evaluated with a 2nd-order central difference along
    each Cartesian direction in turn.
    """
    samples = np.asarray(samples)
    if samples.ndim == 3:
        samples = samples.reshape(-1, 3)

    r = np.linalg.norm(samples, axis=1)
    h = 2e-4

    # base wavefunction values psi(r) = exp(-theta * r)
    psi0 = np.exp(-theta * r)

    # vectorised Laplacian via central differences, one Cartesian axis at a time
    lap = np.zeros_like(r)
    for dim in range(3):
        plus = samples.copy()
        minus = samples.copy()

        plus[:, dim] += h
        minus[:, dim] -= h

        r_plus = np.linalg.norm(plus, axis=1)
        r_minus = np.linalg.norm(minus, axis=1)

        psi_plus = np.exp(-theta * r_plus)
        psi_minus = np.exp(-theta * r_minus)

        lap += (psi_plus - 2.0 * psi0 + psi_minus) / h**2

    # avoid r = 0
    mask = r > 0
    local_energy = np.zeros_like(r)
    local_energy[mask] = -0.5 * lap[mask] / psi0[mask] - 1.0 / r[mask]

    return local_energy[mask].mean()


def monte_carlo_minimisation(theta_step, pdf, iterations, T, theta0,
                              n_walkers=200, pos_steps=500, pos_step_size=1.1):
    """
    Simulated annealing over the variational parameter theta.

    At each iteration, a new theta is proposed and its energy is
    estimated via a fresh multi-walker Metropolis run. The proposal is
    always accepted if it lowers the energy, and accepted with
    probability exp(-delta_E / T) otherwise, allowing occasional uphill
    moves so the search can escape local minima. T is reduced by a
    factor of 0.6 every 10 iterations.

    Returns
    -------
    theta : float
        Final (optimised) value of theta.
    acceptance_rate : float
        Percentage of theta proposals accepted.
    theta_history, energy_history : ndarray
        Theta and energy at every iteration, for convergence plots.
    """
    theta = float(theta0)
    u = np.random.rand(2 * iterations)
    n = 0

    # initial energy at starting theta
    samples, _ = metropolis_3d_multi(pos_step_size, pdf, pos_steps, n_walkers, theta)
    E = energy(samples, theta)

    accepted_number_mc = 0
    theta_history = [theta]
    energy_history = [E]

    for i in range(iterations):
        # propose new theta
        theta_dash = theta + theta_step * (2 * u[i] - 1)

        # sample positions and estimate energy for the proposed theta
        samples_dash, _ = metropolis_3d_multi(pos_step_size, pdf, pos_steps, n_walkers, theta_dash)
        E_dash = energy(samples_dash, theta_dash)

        delta_E = E_dash - E

        if delta_E > 0:
            alpha = np.exp(-delta_E / T)
            accept = u[2 * i] < alpha
        else:
            accept = True

        if accept:
            theta = theta_dash
            E = E_dash
            accepted_number_mc += 1

        theta_history.append(theta)
        energy_history.append(E)

        # cooling schedule
        n += 1
        if n == 10:
            T *= 0.6
            n = 0

    acceptance_rate = accepted_number_mc / iterations * 100
    return theta, acceptance_rate, np.array(theta_history), np.array(energy_history)


def plot_convergence(theta_history, energy_history):
    """Dual-axis plot of theta and energy against iteration number (Fig. 3)."""
    iters = np.arange(theta_history.shape[0])

    plt.figure(figsize=(7, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    lns1 = ax1.plot(iters, theta_history, "-", color="tab:blue", label=r"$\theta$")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(r"$\theta$", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    lns2 = ax2.plot(iters, energy_history, "-", color="tab:red", label="Energy")
    ax2.set_ylabel("Energy", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lns = lns1 + lns2
    labels = [l.get_label() for l in lns]
    plt.legend(lns, labels, loc="best")
    plt.tight_layout()


def main():
    start = time.time()

    theta0 = 2.0

    opt_theta, acceptance_mc, theta_hist, energy_hist = monte_carlo_minimisation(
        theta_step=0.04, pdf=pdf_xyz, iterations=200, T=0.05, theta0=theta0,
        n_walkers=200, pos_steps=1000, pos_step_size=1.1,
    )
    print("Optimized Theta:", opt_theta)
    print(f"Theta acceptance: {acceptance_mc:.1f}%")

    # final, higher-statistics energy estimate at the optimised theta
    samples_optimized, position_acceptance = metropolis_3d_multi(
        step_size=1.1, pdf=pdf_xyz, iterations=2000, n_walkers=200, theta=opt_theta,
    )
    print("Estimated Energy Expectation Value with Optimized Theta:",
          energy(samples_optimized, opt_theta))
    print(f"Position acceptance with optimised theta: {position_acceptance:.1f}%")

    plot_convergence(theta_hist, energy_hist)

    end = time.time()
    print(f"Run time: {end - start:.5f} seconds")

    plt.show()


if __name__ == "__main__":
    main()