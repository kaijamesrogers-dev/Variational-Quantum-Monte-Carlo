import numpy as np
import matplotlib.pyplot as plt
from task_3_1 import lcg_random

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

iterations = 80
n_walkers = 10000

theta1, theta2, theta3 = 1.102459860081999, 0.04286839814990184, 3.489167705406046

samples = metropolis_2protons_multi(
    0.5, pdf_xyz, iterations, n_walkers,
    theta1, theta2, theta3, q_1, q_2, seed=SEED
)

# Discard first 20% as burn-in
burn = samples.shape[0] // 5
samples = samples[burn:]  # shape = ((iterations-burn)*n_walkers, 6)

# Reshape to separate electrons A and B
coords = samples.reshape(-1, 2, 3)    # (N_samples, electron, xyz)
r1 = coords[:, 0, :]                  # electron A positions
r2 = coords[:, 1, :]                  # electron B positions

# Distances of e− from the nearest proton (or treat each proton separately)
r1_q1 = np.linalg.norm(r1 - q_1, axis=1)
r2_q2 = np.linalg.norm(r2 - q_2, axis=1)
r = np.concatenate([r1_q1, r2_q2])

# Histogram of r from samples
fig, ax = plt.subplots()
counts, bins, _ = ax.hist(r, bins=60, density=True, alpha=0.6, label="MC samples")

# Theoretical *shape* (unnormalised), then normalise it to compare shapes
r_grid = np.linspace(0, r.max(), 300)
pdf_shape = r_grid**2 * np.exp(-2 * theta1 * r_grid)

# Normalise to area 1 so we can put it on the same axes:
pdf_shape /= np.trapz(pdf_shape, r_grid)

ax.plot(r_grid, pdf_shape, lw=2, label=r"$r^2 e^{-2\theta_1 r}$ shape (no Jastrow)")
ax.set_xlabel("r (distance to proton)")
ax.set_ylabel("Radial probability density")
ax.legend()
plt.show()

#The code seems to get significatly worse when you go ucnder 80 iterations