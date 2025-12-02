import time
start = time.time()
#------------------

import numpy as np
from task_3_1 import lcg_random
import matplotlib.pyplot as plt


SEED = 42
THETA_INITIAL = 0.5
q_1 = np.array([0.7, 0.0, 0.0])  # position of proton 1
q_2 = np.array([- 0.7, 0.0, 0.0])  # position of proton 2
q = np.array([q_1, q_2])



def psi_h2(x_a, y_a, z_a, x_b, y_b, z_b, theta1, theta2, theta3, q1, q2):
    """
    Trial wavefunction ω(r1, r2; θ1, θ2, θ3) for the H₂ molecule.

    This version is vectorised over the electron coordinates:
    x_a, y_a, ... can be scalars OR 1D NumPy arrays of shape (n_walkers,).

    Parameters
    ----------
    x_a, y_a, z_a : float or ndarray
        Coordinates of electron A.
    x_b, y_b, z_b : float or ndarray
        Coordinates of electron B.
    theta1, theta2, theta3 : float
        Variational parameters (see project equation 18).
    q1, q2 : array_like
        Positions of the two protons, shape (3,).

    Returns
    -------
    psi : float or ndarray
        Wavefunction value(s). If inputs are arrays, output has shape (n_walkers,).
    """
    # Stack coordinates into (..., 3) arrays
    r1 = np.stack((x_a, y_a, z_a), axis=-1)  # shape (..., 3)
    r2 = np.stack((x_b, y_b, z_b), axis=-1)  # shape (..., 3)

    q1 = np.asarray(q1)
    q2 = np.asarray(q2)

    # Distances to nuclei (axis=-1 works for scalars or arrays)
    r1_q1 = np.linalg.norm(r1 - q1, axis=-1)
    r1_q2 = np.linalg.norm(r1 - q2, axis=-1)
    r2_q1 = np.linalg.norm(r2 - q1, axis=-1)
    r2_q2 = np.linalg.norm(r2 - q2, axis=-1)

    # Electron–electron distance
    r12 = np.linalg.norm(r1 - r2, axis=-1)

    # Slater-like part (the bracket in eq. 18)
    slater = (
        np.exp(-theta1 * (r1_q1 + r2_q2)) +
        np.exp(-theta1 * (r1_q2 + r2_q1))
    )

    # Jastrow factor exp(-θ2 / (1 + θ3 r12))
    jastrow = np.exp(-theta2 / (1.0 + theta3 * r12))

    return slater * jastrow


def pdf_xyz(x_a, y_a, z_a, x_b, y_b, z_b, theta1, theta2, theta3, q1, q2):
    """Probability density |ω|² for the H₂ trial wavefunction.

    Works with scalar inputs or 1D arrays (vectorised over walkers).
    """
    psi = psi_h2(x_a, y_a, z_a, x_b, y_b, z_b, theta1, theta2, theta3, q1, q2)
    return psi**2


def _propose_step(current, step_vec):
    """Generate a proposed position by adding the random step."""
    return current + step_vec


def metropolis_2protons_multi(step_size, pdf, iterations, n_walkers,
                              theta1, theta2, theta3, q_1, q_2, seed=SEED):
    """
    Run n_walkers independent 3D Metropolis–Hastings chains for two electrons
    in parallel, using a vectorised pdf.

    Parameters
    ----------
    step_size : float
    pdf       : callable
        Vectorised pdf:
        pdf(x_a, y_a, z_a, x_b, y_b, z_b, theta1, theta2, theta3, q_1, q_2)
        where x_a, ... can be arrays of shape (n_walkers,).
    iterations : int
        Number of Metropolis steps per walker.
    n_walkers : int
        Number of independent chains to run in parallel.
    theta1, theta2, theta3 : float
        Variational parameters.
    q_1, q_2 : array_like
        Proton positions.
    seed : int
        Seed for the LCG RNG.

    Returns
    -------
    samples : ndarray
        Shape (iterations * n_walkers, 6), where the last axis is
        [x_a, y_a, z_a, x_b, y_b, z_b].
    """
    # 6 * n_walkers for the initial point + (6 * n_walkers for proposal
    # + n_walkers for accept) per step
    n_uniform = n_walkers * (6 + 7 * (iterations - 1))
    u = lcg_random(seed, n_uniform)
    idx = 0

    samples = np.zeros((iterations, n_walkers, 6))

# --- New initialisation: start electrons near each proton ----
    offset_scale = 0.5    # how far from each proton to initialise (tunable)

    # Use 6 * n_walkers uniforms for initial offsets
    u_init = u[idx:idx + 6 * n_walkers].reshape(n_walkers, 6)
    idx += 6 * n_walkers

    # Convert uniforms in [0,1] to offsets in [-offset_scale, +offset_scale]
    offsets = offset_scale * (2.0 * u_init - 1.0)

    # Electron A initial positions near proton q_1
    samples[0, :, 0:3] = q_1 + offsets[:, 0:3]

    # Electron B initial positions near proton q_2
    samples[0, :, 3:6] = q_2 + offsets[:, 3:6]


    for i in range(1, iterations):
        current = samples[i - 1]          # shape (n_walkers, 6)

        # Proposal steps shaped (n_walkers, 6) with each coordinate
        # uniformly in [-step_size, step_size]
        proposal_steps = step_size * (2 * u[idx:idx + 6 * n_walkers].reshape(n_walkers, 6) - 1)
        idx += 6 * n_walkers

        proposal = _propose_step(current, proposal_steps)

        # Acceptance uniforms for all walkers at this step
        u_accept = u[idx:idx + n_walkers]
        idx += n_walkers

        # Split coordinates for vectorised pdf evaluation
        x_a_c, y_a_c, z_a_c = current[:, 0], current[:, 1], current[:, 2]
        x_b_c, y_b_c, z_b_c = current[:, 3], current[:, 4], current[:, 5]

        x_a_p, y_a_p, z_a_p = proposal[:, 0], proposal[:, 1], proposal[:, 2]
        x_b_p, y_b_p, z_b_p = proposal[:, 3], proposal[:, 4], proposal[:, 5]

        # Vectorised pdf for all walkers at once
        p_current = pdf(x_a_c, y_a_c, z_a_c,
                        x_b_c, y_b_c, z_b_c,
                        theta1, theta2, theta3, q_1, q_2)

        p_prop = pdf(x_a_p, y_a_p, z_a_p,
                     x_b_p, y_b_p, z_b_p,
                     theta1, theta2, theta3, q_1, q_2)

        # Compute acceptance probabilities
        alpha = np.ones(n_walkers)
        valid = p_current > 0
        alpha[valid] = np.minimum(1.0, p_prop[valid] / p_current[valid])

        accept = u_accept < alpha

        # Broadcast accept mask over the last axis to update accepted walkers only
        samples[i, :, :] = np.where(accept[:, None], proposal, current)

    # OPTION A: flatten all walkers into one big sample set
    return samples.reshape(iterations * n_walkers, 6)


def energy(samples, theta1, theta2, theta3, q):
    """Estimate the energy expectation value using vectorised operations."""
    coords = samples.reshape(-1, 2, 3)  # (n_samples, electron, xyz)
    r1 = coords[:, 0]
    r2 = coords[:, 1]

    q12 = np.linalg.norm(q[0] - q[1])
    h = 1e-4
    eps = 1e-12  # avoids division spikes if a walker hits a nucleus

    # Base wavefunction values for all samples
    psi_base = psi_h2(
        r1[:, 0], r1[:, 1], r1[:, 2],
        r2[:, 0], r2[:, 1], r2[:, 2],
        theta1, theta2, theta3, q[0], q[1]
    )

    # Vectorised central-difference Laplacian over both electrons and xyz
    laplacian = np.zeros_like(psi_base)
    for e_idx in range(2):
        for dim in range(3):
            coords_plus = coords.copy()
            coords_minus = coords.copy()
            coords_plus[:, e_idx, dim] += h
            coords_minus[:, e_idx, dim] -= h

            psi_plus = psi_h2(
                coords_plus[:, 0, 0], coords_plus[:, 0, 1], coords_plus[:, 0, 2],
                coords_plus[:, 1, 0], coords_plus[:, 1, 1], coords_plus[:, 1, 2],
                theta1, theta2, theta3, q[0], q[1]
            )
            psi_minus = psi_h2(
                coords_minus[:, 0, 0], coords_minus[:, 0, 1], coords_minus[:, 0, 2],
                coords_minus[:, 1, 0], coords_minus[:, 1, 1], coords_minus[:, 1, 2],
                theta1, theta2, theta3, q[0], q[1]
            )

            laplacian += psi_plus - 2.0 * psi_base + psi_minus

    laplacian /= h ** 2

    # Potential energy terms in bulk
    r1_q1 = np.linalg.norm(r1 - q[0], axis=1) + eps
    r1_q2 = np.linalg.norm(r1 - q[1], axis=1) + eps
    r2_q1 = np.linalg.norm(r2 - q[0], axis=1) + eps
    r2_q2 = np.linalg.norm(r2 - q[1], axis=1) + eps
    r12 = np.linalg.norm(r1 - r2, axis=1) + eps

    potential = (
        -1.0 / r1_q1
        -1.0 / r1_q2
        -1.0 / r2_q1
        -1.0 / r2_q2
        + 1.0 / q12
        + 1.0 / r12
    )

    local_E = -(0.5 / psi_base) * laplacian + potential
    return np.mean(local_E)


def monte_carlo_minimisation_2protons(step_size, pdf, iterations, T,
                                      theta=THETA_INITIAL, seed=SEED):
    # Draw ONE chain at the start, using the initial theta
    """Simple simulated annealing on θ₁, θ₂, θ₃ using a reused chain."""
    theta1 = theta
    theta2 = theta
    theta3 = theta

    samples = metropolis_2protons_multi(step_size, pdf, 1000, 50,
                                  theta1, theta2, theta3, q_1, q_2, seed)

    E = energy(samples, theta1, theta2, theta3, q)

    # Use different random numbers just for parameter proposals
    u = lcg_random(seed + 1, 3 * iterations)
    n = 0
    a = 0

    for i in range(iterations):

        du1, du2, du3 = 2 * u[3 * i:3 * i + 3] - 1

        theta1_dash = theta1 + step_size * du1
        theta2_dash = theta2 + step_size * du2
        theta3_dash = theta3 + step_size * du3

        if not (0.0 < theta1_dash < 5.0 and
            0.0 < theta2_dash < 5.0 and
            0.0 <= theta3_dash < 5.0):
            continue


        samples_dash = metropolis_2protons_multi(step_size, pdf, 1000, 50,
                                  theta1_dash, theta2_dash, theta3_dash, q_1, q_2, seed)
        E_dash = energy(samples_dash, theta1_dash, theta2_dash, theta3_dash, q)

        delta_E = E_dash - E

        if delta_E > 0:
            alpha = np.exp(- delta_E / T)
            rand = np.random.rand()
            if rand < alpha:
                theta1, theta2, theta3 = theta1_dash, theta2_dash, theta3_dash
                E = E_dash
        else:
            theta1, theta2, theta3 = theta1_dash, theta2_dash, theta3_dash
            E = E_dash
    
        a += 1
        print(f"Theta values {a}:", theta1, theta2, theta3)
        print(f"T values {a}:", T)
        n += 1
        
        if n == 5:
            T *= 0.8
            n = 0

    return theta1, theta2, theta3


def plot_h2_pdf_histogram(theta1, theta2, theta3, 
                          q_1, q_2,
                          n_samples=1000, n_walkers=10,
                          step_size=0.5,
                          bins=120):
    """
    Plot a 2D histogram of electron positions (x vs z) for the H2 molecule.
    Samples are drawn from the Metropolis distribution |psi|^2.

    Parameters
    ----------
    theta1, theta2, theta3 : float
        Optimised variational parameters.
    q_1, q_2 : array_like
        Proton positions.
    n_samples : int
        Number of Metropolis samples.
    step_size : float
        Proposal step size for Metropolis.
    bins : int
        Number of bins for the 2D histogram.
    """

    # ---- 1. Generate samples from |psi|² ------------------------
    samples = metropolis_2protons_multi(step_size, pdf_xyz,
                                  n_samples, n_walkers,
                                  theta1, theta2, theta3,
                                  q_1, q_2, SEED)

    # samples[:, :3] → electron A
    # samples[:, 3:] → electron B
    r1 = samples[:, :3]
    r2 = samples[:, 3:]

    # ---- 2. Convert to x,z coordinates (ignore y) ---------------
    x_vals = np.concatenate([r1[:, 0], r2[:, 0]])   # electron A + B
    z_vals = np.concatenate([r1[:, 2], r2[:, 2]])

    # ---- 3. Build 2D histogram ---------------------------------
    plt.figure(figsize=(6, 5))

    plt.hist2d(x_vals, z_vals, bins=bins, density=True,
               cmap="inferno")

    plt.xlabel("x (a.u.)")
    plt.ylabel("z (a.u.)")
    plt.title("2D Histogram of Electron Probability Density for H₂")

    #plt.xlim(-5, 5)
    #plt.ylim(-5, 5)

    # Colour bar
    plt.colorbar(label="Probability Density")

    # Mark proton positions in the x–z plane
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
        q_1 = np.array([i, 0.0, 0.0])  # position of proton 1
        q_2 = np.array([- i, 0.0, 0.0])  # position of proton 2
        q = np.array([q_1, q_2])
        theta1, theta2, theta3 = monte_carlo_minimisation_2protons(step_size, pdf, iterations, T, THETA_INITIAL, SEED)
        energy_estimate = energy(metropolis_2protons_multi(step_size, pdf, 1000, 10,
                                  theta1, theta2, theta3, q_1, q_2, SEED), theta1, theta2, theta3, q)
        
        energy_estimates.append(energy_estimate)
        bond_lengths.append(i)

    plt.figure(figsize=(6, 5))
    plt.plot(bond_lengths, energy_estimates, marker='o')
    plt.xlabel("Bond Length (a.u.)")
    plt.ylabel("Energy Estimate (a.u.)")
    plt.title("Energy Estimate vs Bond Length for H₂ Molecule")
    plt.grid(True)
    plt.show()


theta1, theta2, theta3 = monte_carlo_minimisation_2protons(0.5, pdf_xyz, 100, 0.5, THETA_INITIAL, SEED)
print("Optimized theta values for H2 molecule:", theta1, theta2, theta3)

energy_estimate = energy(metropolis_2protons_multi(0.5, pdf_xyz, 1000, 100,
                                  theta1, theta2, theta3, q_1, q_2, SEED), theta1, theta2, theta3, q)
print("Estimated Energy Expectation Value for H2 molecule:", energy_estimate)

plot_h2_pdf_histogram(theta1, theta2, theta3, q_1, q_2)
#bond_length(0.5, pdf_xyz, 100, 0.5, 3, 13, THETA_INITIAL, SEED)
#-------------------------------------------
end = time.time()
print(f"Run time: {end - start:.5f} seconds")
