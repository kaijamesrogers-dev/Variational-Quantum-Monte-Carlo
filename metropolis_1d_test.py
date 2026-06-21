"""
Metropolis algorithm test in 1D: harmonic oscillator.

Applies the Metropolis algorithm to sample the ground-state probability
density of the 1D harmonic oscillator, then uses the samples to estimate
the ground- and first-excited-state energies via the Monte Carlo energy
estimator. This validates the sampling procedure before extending it to
3D (hydrogen atom) and 6D (hydrogen molecule).
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


def pdf0(r):
    """Ground-state probability density of the harmonic oscillator, rho_0(x)."""
    return np.exp(-r**2) / np.sqrt(np.pi)


def metropolis(step_size, pdf, iterations):
    """
    Single-chain 1D Metropolis sampler.

    Parameters
    ----------
    step_size : float
        Half-width of the uniform proposal distribution.
    pdf : callable
        Target probability density (need not be normalised).
    iterations : int
        Number of steps in the chain.

    Returns
    -------
    x : ndarray, shape (iterations,)
        Sampled positions.
    acceptance_pct : float
        Percentage of proposed moves that were accepted.
    """
    u = np.random.rand(iterations * 2 + 1)
    accepted_count = 0

    x = np.zeros(iterations)
    x[0] = (u[0] - 0.5) * 2  # initial point in [-1, 1]

    for i in range(1, iterations):
        x_current = x[i - 1]

        # proposal step
        u_step = u[2 * i - 1]
        u_accept = u[2 * i]
        x_proposed = x_current + step_size * (2 * u_step - 1)

        # Metropolis acceptance ratio
        alpha = min(1.0, pdf(x_proposed) / pdf(x_current))
        accept = u_accept < alpha
        accepted_count += accept.sum()
        x[i] = np.where(accept, x_proposed, x_current)

    return x, accepted_count / iterations * 100


def metropolis_multi(step_size, pdf, iterations, n_walkers):
    """
    Run n_walkers independent 1D Metropolis chains in parallel.

    Returns
    -------
    samples : 1D ndarray
        Flattened array of shape (iterations * n_walkers,).
    acceptance_frac : float
        Fraction of proposed moves that were accepted.
    """
    # pre-generate all random numbers
    u = np.random.rand(n_walkers * (1 + 2 * (iterations - 1)))
    idx = 0

    x = np.zeros((iterations, n_walkers))

    # initial positions in [-1, 1]
    x[0] = (u[idx:idx + n_walkers] - 0.5) * 2.0
    idx += n_walkers

    accepted_count = 0

    for i in range(1, iterations):
        x_current = x[i - 1]

        # proposal uniforms and accept uniforms for all walkers
        u_step = u[idx:idx + n_walkers]
        idx += n_walkers
        u_accept = u[idx:idx + n_walkers]
        idx += n_walkers

        # propose new positions
        x_proposed = x_current + step_size * (2 * u_step - 1)

        # vectorised pdf evaluation
        p_current = pdf(x_current)
        p_prop = pdf(x_proposed)

        # avoid division by zero
        alpha = np.ones(n_walkers)
        valid = p_current > 0
        alpha[valid] = np.minimum(1.0, p_prop[valid] / p_current[valid])

        accept = u_accept < alpha
        accepted_count += accept.sum()
        x[i] = np.where(accept, x_proposed, x_current)

    return x.reshape(iterations * n_walkers), accepted_count / (iterations * n_walkers)


def phi0(x):
    """Ground-state harmonic oscillator wavefunction, psi_0(x)."""
    return np.exp(-x**2 / 2)


def phi1(x):
    """First-excited-state harmonic oscillator wavefunction, psi_1(x)."""
    return 2 * x * np.exp(-x**2 / 2)


def estimate_energy(wavefunction, samples, h=0.0002):
    """
    Monte Carlo estimate of <H> for the 1D harmonic oscillator, given
    samples drawn from |wavefunction|^2.

    Uses the local energy E_l(x) = -1/2 * psi''(x)/psi(x) + 1/2 * x^2,
    averaged over all samples (Eq. 5 in the report).
    """
    laplacian = w2_2nd_general(wavefunction, samples, h)
    local_energy = -0.5 * laplacian / wavefunction(samples) + 0.5 * samples**2
    return np.mean(local_energy)


def main():
    start = time.time()

    N = 1_000_000
    samples, percentage = metropolis(2, pdf0, N)
    print(f"Acceptance rate: {percentage:.1f}%")

    # histogram of samples vs. the analytical PDF (Fig. 2)
    plt.hist(samples, bins=30, density=True, alpha=0.6, label="Samples")
    r = np.arange(-5, 5, 0.1)
    plt.plot(r, pdf0(r), label="PDF", color="red")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()

    # energy expectation values for the ground and first excited states
    energy_0 = estimate_energy(phi0, samples)
    print("Estimated Energy Expectation Value for n = 0:", energy_0)

    energy_1 = estimate_energy(phi1, samples)
    print("Estimated Energy Expectation Value for n = 1:", energy_1)

    end = time.time()
    print(f"Run time: {end - start:.5f} seconds")

    plt.show()


if __name__ == "__main__":
    main()