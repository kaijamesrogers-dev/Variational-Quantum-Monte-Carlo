"""Variational Monte Carlo utilities for the H2 molecule.

This module provides a faster version of the reference script by
vectorising the expensive Monte Carlo and energy-evaluation steps.
"""

import matplotlib.pyplot as plt
import numpy as np


SEED = 42
THETA_INITIAL = 0.5
q_1 = np.array([0.7, 0.0, 0.0])
q_2 = np.array([-0.7, 0.0, 0.0])
q = np.array([q_1, q_2])


# ---------------------------------------------------------------------------
# Random number helper
# ---------------------------------------------------------------------------

def lcg_random(seed: int, n: int, modulus: int = 2**31 - 1,
               a: int = 1103515245, c: int = 12345) -> np.ndarray:
    """Lightweight linear congruential generator matching task_3_1.lcg_random."""
    nums = np.empty(n, dtype=float)
    state = seed
    for i in range(n):
        state = (a * state + c) % modulus
        nums[i] = state / modulus
    return nums


# ---------------------------------------------------------------------------
# Wavefunction and pdf
# ---------------------------------------------------------------------------

def psi_h2(x_a, y_a, z_a, x_b, y_b, z_b, theta1, theta2, theta3, q1, q2):
    """Trial wavefunction ω(r1, r2; θ1, θ2, θ3) for the H₂ molecule."""
    r1 = np.stack((x_a, y_a, z_a), axis=-1)
    r2 = np.stack((x_b, y_b, z_b), axis=-1)

    q1 = np.asarray(q1)
    q2 = np.asarray(q2)

    r1_q1 = np.linalg.norm(r1 - q1, axis=-1)
    r1_q2 = np.linalg.norm(r1 - q2, axis=-1)
    r2_q1 = np.linalg.norm(r2 - q1, axis=-1)
    r2_q2 = np.linalg.norm(r2 - q2, axis=-1)

    r12 = np.linalg.norm(r1 - r2, axis=-1)

    slater = (
        np.exp(-theta1 * (r1_q1 + r2_q2)) +
        np.exp(-theta1 * (r1_q2 + r2_q1))
    )

    jastrow = np.exp(-theta2 / (1.0 + theta3 * r12))

    return slater * jastrow


def pdf_xyz(x_a, y_a, z_a, x_b, y_b, z_b, theta1, theta2, theta3, q1, q2):
    """Probability density |ω|² for the H₂ trial wavefunction."""
    psi = psi_h2(x_a, y_a, z_a, x_b, y_b, z_b, theta1, theta2, theta3, q1, q2)
    return psi ** 2


# ---------------------------------------------------------------------------
# Metropolis sampler
# ---------------------------------------------------------------------------

def _propose_step(current: np.ndarray, step_vec: np.ndarray) -> np.ndarray:
    """Generate a proposed position by adding the random step."""
    return current + step_vec


def metropolis_2protons_multi(step_size: float, pdf, iterations: int, n_walkers: int,
                              theta1: float, theta2: float, theta3: float,
                              q_1: np.ndarray, q_2: np.ndarray, seed: int = SEED,
                              initial_positions: np.ndarray | None = None,
                              return_chain: bool = False,
                              burn_in: int = 0) -> np.ndarray:
    """Run n_walkers independent 3D Metropolis–Hastings chains in parallel.

    Parameters
    ----------
    initial_positions:
        Optional array of shape (n_walkers, 6) to warm start the chain.
    return_chain:
        If True, return both the flattened chain and the final walker positions.
    burn_in:
        Number of initial iterations to discard when returning the flattened
        samples. The final walker positions always come from the last
        iteration regardless of burn-in.
    """
    if burn_in < 0 or burn_in >= iterations:
        raise ValueError("burn_in must be between 0 and iterations - 1")

    n_init_draws = 0 if initial_positions is not None else 6 * n_walkers
    n_uniform = n_init_draws + (iterations - 1) * 7 * n_walkers
    u = lcg_random(seed, n_uniform)
    idx = 0

    samples = np.zeros((iterations, n_walkers, 6))

    if initial_positions is None:
        u_init = u[idx:idx + 6 * n_walkers].reshape(n_walkers, 6)
        idx += 6 * n_walkers
        offsets = 0.5 * (2.0 * u_init - 1.0)
        samples[0, :, 0:3] = q_1 + offsets[:, 0:3]
        samples[0, :, 3:6] = q_2 + offsets[:, 3:6]
    else:
        samples[0] = initial_positions

    for i in range(1, iterations):
        current = samples[i - 1]

        proposal_steps = step_size * (
            2 * u[idx:idx + 6 * n_walkers].reshape(n_walkers, 6) - 1
        )
        idx += 6 * n_walkers
        proposal = _propose_step(current, proposal_steps)

        u_accept = u[idx:idx + n_walkers]
        idx += n_walkers

        combined = np.concatenate([current, proposal], axis=0)
        x_a, y_a, z_a = combined[:, 0], combined[:, 1], combined[:, 2]
        x_b, y_b, z_b = combined[:, 3], combined[:, 4], combined[:, 5]
        p_all = pdf(x_a, y_a, z_a, x_b, y_b, z_b,
                    theta1, theta2, theta3, q_1, q_2)
        p_current, p_prop = np.split(p_all, 2)

        alpha = np.ones(n_walkers)
        valid = p_current > 0
        alpha[valid] = np.minimum(1.0, p_prop[valid] / p_current[valid])

        accept = u_accept < alpha
        samples[i] = np.where(accept[:, None], proposal, current)

    flattened = samples[burn_in:].reshape((iterations - burn_in) * n_walkers, 6)
    if return_chain:
        return flattened, samples[-1]
    return flattened


# ---------------------------------------------------------------------------
# Energy estimator
# ---------------------------------------------------------------------------

def energy(samples: np.ndarray, theta1: float, theta2: float, theta3: float, q: np.ndarray) -> float:
    """Estimate the energy expectation value using vectorised operations."""
    coords = samples.reshape(-1, 2, 3)
    r1 = coords[:, 0]
    r2 = coords[:, 1]

    q12 = np.linalg.norm(q[0] - q[1])
    h = 1e-4
    eps = 1e-12

    psi_base = psi_h2(
        r1[:, 0], r1[:, 1], r1[:, 2],
        r2[:, 0], r2[:, 1], r2[:, 2],
        theta1, theta2, theta3, q[0], q[1]
    )

    # Vectorised central-difference Laplacian over both electrons and xyz.
    shift_vectors = np.eye(6).reshape(6, 2, 3)
    coords_plus = coords[:, None, :, :] + h * shift_vectors
    coords_minus = coords[:, None, :, :] - h * shift_vectors

    def _psi_for_shifted(arr: np.ndarray) -> np.ndarray:
        flattened = arr.reshape(-1, 2, 3)
        return psi_h2(flattened[:, 0, 0], flattened[:, 0, 1], flattened[:, 0, 2],
                      flattened[:, 1, 0], flattened[:, 1, 1], flattened[:, 1, 2],
                      theta1, theta2, theta3, q[0], q[1]).reshape(arr.shape[0], -1)

    psi_plus = _psi_for_shifted(coords_plus)
    psi_minus = _psi_for_shifted(coords_minus)

    laplacian = np.sum(psi_plus + psi_minus - 2.0 * psi_base[:, None], axis=1) / h ** 2

    r1_q1 = np.linalg.norm(r1 - q[0], axis=1) + eps
    r1_q2 = np.linalg.norm(r1 - q[1], axis=1) + eps
    r2_q1 = np.linalg.norm(r2 - q[0], axis=1) + eps
    r2_q2 = np.linalg.norm(r2 - q[1], axis=1) + eps
    r12 = np.linalg.norm(r1 - r2, axis=1) + eps

    potential = (
        -1.0 / r1_q1
        - 1.0 / r1_q2
        - 1.0 / r2_q1
        - 1.0 / r2_q2
        + 1.0 / q12
        + 1.0 / r12
    )

    local_E = -(0.5 / psi_base) * laplacian + potential
    return float(np.mean(local_E))


# ---------------------------------------------------------------------------
# Simulated annealing over theta parameters
# ---------------------------------------------------------------------------

def monte_carlo_minimisation_2protons(step_size: float, pdf, iterations: int, T: float,
                                      theta: float = THETA_INITIAL, seed: int = SEED,
                                      chain_iterations: int = 100, n_walkers: int = 500,
                                      burn_in: int | None = None):
    """Simple simulated annealing on θ₁, θ₂, θ₃ using reusable chains."""
    theta1 = theta2 = theta3 = theta

    burn_in_steps = chain_iterations // 5 if burn_in is None else burn_in

    current_samples, last_positions = metropolis_2protons_multi(
        step_size, pdf, chain_iterations, n_walkers,
        theta1, theta2, theta3, q_1, q_2, seed,
        return_chain=True, burn_in=burn_in_steps
    )
    E = energy(current_samples, theta1, theta2, theta3, q)

    rng = np.random.default_rng(seed + 1)
    cool_counter = 0

    for _ in range(iterations):
        du1, du2, du3 = 2 * rng.random(3) - 1

        theta1_dash = theta1 + step_size * du1
        theta2_dash = theta2 + step_size * du2
        theta3_dash = theta3 + step_size * du3

        if not (0.0 < theta1_dash < 5.0 and 0.0 < theta2_dash < 5.0 and 0.0 <= theta3_dash < 5.0):
            continue

        candidate_samples, candidate_last = metropolis_2protons_multi(
            step_size, pdf, chain_iterations, n_walkers,
            theta1_dash, theta2_dash, theta3_dash, q_1, q_2,
            seed, initial_positions=last_positions,
            return_chain=True, burn_in=burn_in_steps
        )
        E_dash = energy(candidate_samples, theta1_dash, theta2_dash, theta3_dash, q)

        delta_E = E_dash - E
        accept = delta_E <= 0 or rng.random() < np.exp(-delta_E / T)

        if accept:
            theta1, theta2, theta3 = theta1_dash, theta2_dash, theta3_dash
            E = E_dash
            current_samples = candidate_samples
            last_positions = candidate_last

        cool_counter += 1
        if cool_counter == 5:
            T *= 0.8
            cool_counter = 0

    return theta1, theta2, theta3


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_h2_pdf_histogram(theta1, theta2, theta3,
                          q_1, q_2,
                          n_samples=1000, n_walkers=10,
                          step_size=0.5,
                          bins=120):
    """Plot a 2D histogram of electron positions (x vs z) for the H2 molecule."""
    samples = metropolis_2protons_multi(step_size, pdf_xyz,
                                        n_samples, n_walkers,
                                        theta1, theta2, theta3,
                                        q_1, q_2, SEED,
                                        burn_in=n_samples // 5)

    r1 = samples[:, :3]
    r2 = samples[:, 3:]

    x_vals = np.concatenate([r1[:, 0], r2[:, 0]])
    z_vals = np.concatenate([r1[:, 2], r2[:, 2]])

    plt.figure(figsize=(6, 5))

    plt.hist2d(x_vals, z_vals, bins=bins, density=True,
               cmap="inferno")

    plt.xlabel("x (a.u.)")
    plt.ylabel("z (a.u.)")
    plt.title("2D Histogram of Electron Probability Density for H₂")

    plt.colorbar(label="Probability Density")

    plt.scatter([q_1[0], q_2[0]],
                [q_1[2], q_2[2]],
                c="cyan", s=80, marker="x", label="Protons")

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def bond_length(step_size, pdf, iterations, T, r_min, r_max, THETA_INITIAL, SEED):
    energy_estimates = []
    bond_lengths = []

    for i in np.arange(r_min, r_max, 0.2):
        q_1 = np.array([i, 0.0, 0.0])
        q_2 = np.array([-i, 0.0, 0.0])
        q = np.array([q_1, q_2])
        theta1, theta2, theta3 = monte_carlo_minimisation_2protons(step_size, pdf, iterations, T, THETA_INITIAL, SEED)
        energy_estimate = energy(metropolis_2protons_multi(step_size, pdf, 50, 2000,
                                                          theta1, theta2, theta3, q_1, q_2, SEED,
                                                          burn_in=110),
                                 theta1, theta2, theta3, q)

        energy_estimates.append(energy_estimate)
        bond_lengths.append(i)

    plt.figure(figsize=(6, 5))
    plt.plot(bond_lengths, energy_estimates, marker='o')
    plt.xlabel("Bond Length (a.u.)")
    plt.ylabel("Energy Estimate (a.u.)")
    plt.title("Energy Estimate vs Bond Length for H₂ Molecule")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    theta1, theta2, theta3 = monte_carlo_minimisation_2protons(
        0.5, pdf_xyz, iterations=100, T=0.5, theta=THETA_INITIAL, seed=SEED,
        chain_iterations=50, n_walkers=100, burn_in=10
    )
    print("Optimized theta values for H2 molecule:", theta1, theta2, theta3)

    energy_estimate = energy(
        metropolis_2protons_multi(0.5, pdf_xyz, 50, 2000,
                                  theta1, theta2, theta3, q_1, q_2, SEED,
                                  burn_in=10),
        theta1, theta2, theta3, q
    )
    print("Estimated Energy Expectation Value for H2 molecule:", energy_estimate)