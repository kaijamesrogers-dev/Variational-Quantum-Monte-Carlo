import time
start = time.time()
#------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

THETA_INITIAL = 0.5
q_1 = np.array([0.6611475, 0.0, 0.0])  # position of proton 1
q_2 = np.array([- 0.6611475, 0.0, 0.0])  # position of proton 2
q = np.array([q_1, q_2])
E_single = -0.5



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


def metropolis_2protons_multi(step_size, pdf, n_walkers, theta1, theta2, theta3, q_1, q_2, iterations=100):

    n_uniform = n_walkers * (6 + 7 * (iterations - 1))
    u = np.random.rand(n_uniform)
    idx = 0

    samples = np.zeros((iterations, n_walkers, 6))

    # --- Acceptance counters ---
    total_moves = (iterations - 1) * n_walkers
    accepted_moves = 0

    # Initial positions
    offset_scale = 0.5
    u_init = u[idx:idx + 6 * n_walkers].reshape(n_walkers, 6)
    idx += 6 * n_walkers
    offsets = offset_scale * (2.0 * u_init - 1.0)

    samples[0, :, 0:3] = q_1 + offsets[:, 0:3]
    samples[0, :, 3:6] = q_2 + offsets[:, 3:6]

    for i in range(1, iterations):
        current = samples[i - 1]

        proposal_steps = step_size * (
            2 * u[idx:idx + 6 * n_walkers].reshape(n_walkers, 6) - 1
        )
        idx += 6 * n_walkers

        proposal = current + proposal_steps

        u_accept = u[idx:idx + n_walkers]
        idx += n_walkers

        x_a_c, y_a_c, z_a_c = current[:, 0], current[:, 1], current[:, 2]
        x_b_c, y_b_c, z_b_c = current[:, 3], current[:, 4], current[:, 5]

        x_a_p, y_a_p, z_a_p = proposal[:, 0], proposal[:, 1], proposal[:, 2]
        x_b_p, y_b_p, z_b_p = proposal[:, 3], proposal[:, 4], proposal[:, 5]

        p_current = pdf(x_a_c, y_a_c, z_a_c,
                        x_b_c, y_b_c, z_b_c,
                        theta1, theta2, theta3, q_1, q_2)

        p_prop = pdf(x_a_p, y_a_p, z_a_p,
                     x_b_p, y_b_p, z_b_p,
                     theta1, theta2, theta3, q_1, q_2)

        alpha = np.ones(n_walkers)
        valid = p_current > 0
        alpha[valid] = np.minimum(1.0, p_prop[valid] / p_current[valid])

        accept = u_accept < alpha

        # ---- Count acceptances ----
        accepted_moves += np.sum(accept)

        samples[i] = np.where(accept[:, None], proposal, current)

    burn_in = max(1, int(0.2 * iterations))
    samples_post = samples[burn_in:, :, :]

    acceptance_rate = accepted_moves / total_moves

    print(f"Acceptance rate: {100*acceptance_rate:.2f}% "
          f"({accepted_moves}/{total_moves})")

    return samples_post.reshape(-1, 6)


def energy(samples, theta1, theta2, theta3, q):
    """Estimate the energy expectation value using vectorised operations."""
    coords = samples.reshape(-1, 2, 3)  # (n_samples, electron, xyz)
    r1 = coords[:, 0]
    r2 = coords[:, 1]

    q12 = np.linalg.norm(q[0] - q[1])
    h = 10**(-3.5)
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


def monte_carlo_minimisation_2protons(step_size_r, step_size_theta, pdf, iterations, T, q_1, q_2, q, theta=THETA_INITIAL):
    theta1 = theta
    theta2 = theta
    theta3 = theta

    samples = metropolis_2protons_multi(step_size_r, pdf, 1000,
                                        theta1, theta2, theta3, q_1, q_2)
    E = energy(samples, theta1, theta2, theta3, q)

    u = np.random.rand(3 * iterations)
    n = 0
    a = 0
    accepted_theta_number = 0

    for i in range(iterations):
        du1, du2, du3 = 2 * u[3*i:3*i+3] - 1

        theta1_dash = theta1 + step_size_theta * du1
        theta2_dash = theta2 + step_size_theta * du2
        theta3_dash = theta3 + step_size_theta * du3

        if not (0.0 < theta1_dash < 5.0 and
                0.0 < theta2_dash < 5.0 and
                0.0 <= theta3_dash < 5.0):
            continue

        samples_dash = metropolis_2protons_multi(step_size_r, pdf, 500,
                                                 theta1_dash, theta2_dash, theta3_dash,
                                                 q_1, q_2)
        E_dash = energy(samples_dash, theta1_dash, theta2_dash, theta3_dash, q)

        delta_E = E_dash - E

        if delta_E > 0:
            alpha = np.exp(- delta_E / T)
            rand = np.random.rand()
            if rand < alpha:
                theta1, theta2, theta3 = theta1_dash, theta2_dash, theta3_dash
                E = E_dash
                accepted_theta_number += 1
        else:
            theta1, theta2, theta3 = theta1_dash, theta2_dash, theta3_dash
            E = E_dash
            accepted_theta_number += 1

        a += 1
        print(f"Theta values {a}:", theta1, theta2, theta3)
        print(f"T values {a}:", T)
        n += 1

        if n == 5:
            T *= 0.5
            n = 0

    return theta1, theta2, theta3, accepted_theta_number / iterations * 100



def plot_h2_pdf_histogram(theta1, theta2, theta3, q_1, q_2, n_walkers=500000, step_size=0.5, bins=600 ):
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
                                 n_walkers,
                                  theta1, theta2, theta3,
                                  q_1, q_2)

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

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    # Colour bar
    plt.colorbar(label="Probability Density")

    # Mark proton positions in the x–z plane
    plt.scatter([q_1[0], q_2[0]],
                [q_1[2], q_2[2]],
                c="cyan", s=80, marker="x", label="Protons")

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def morse_total(r, D, a, r0, E_single):
    """Morse potential in the project form: D(1 - e^{-a(r-r0)})^2 - D + 2E_single."""
    return D * (1.0 - np.exp(-a * (r - r0)))**2 - D + 2.0 * E_single


def bond_length(step_size_r, pdf, iterations, T, r_min, r_max, THETA_INITIAL, E_single):

    energy_estimates = []
    half_separations = []   # this is your 'i' (so physical R = 2*i)

    for i in np.arange(r_min, r_max, 0.1):
        q_1 = np.array([i, 0.0, 0.0])   # proton 1 at +i
        q_2 = np.array([-i, 0.0, 0.0])  # proton 2 at -i
        q   = np.array([q_1, q_2])

        # Optimise thetas for THIS separation
        theta1, theta2, theta3, percentage_theta = monte_carlo_minimisation_2protons(
            step_size_r=step_size_r,
            step_size_theta=0.3,      # tweak as you like
            pdf=pdf,
            iterations=iterations,
            T=T,
            q_1=q_1,
            q_2=q_2,
            q=q,
            theta=THETA_INITIAL
        )
        print(f"Percentage of theta accepted at separation {2*i}: {percentage_theta:.2f}%")
        # High-statistics energy estimate at this separation
        samples = metropolis_2protons_multi(
            step_size_r, pdf,
            n_walkers=1000,
            theta1=theta1, theta2=theta2, theta3=theta3,
            q_1=q_1, q_2=q_2,
            iterations=100
        )
        energy_estimate = energy(samples, theta1, theta2, theta3, q)

        energy_estimates.append(energy_estimate)
        half_separations.append(i)

    # Convert lists to arrays
    half_separations = np.array(half_separations)
    energies = np.array(energy_estimates)

    # Physical bond length r: distance between protons = 2*i
    r_data = 2.0 * half_separations

    # ---- Fit Morse potential in project form ----
    # Rough initial guesses
    E_min_guess = energies.min()
    r0_guess = r_data[np.argmin(energies)]
    D_guess = energies.max() - E_min_guess  # rough well depth
    a_guess = 1.0                           # typical order of magnitude

    p0 = [D_guess, a_guess, r0_guess, E_single]

    # We want to fit only D, a, r0 (E_single is known),
    # so wrap a 3-parameter function for curve_fit:
    def morse_fit_three(r, D, a, r0):
        return morse_total(r, D, a, r0, E_single)

    params, cov = curve_fit(morse_fit_three, r_data, energies,
                            p0=[D_guess, a_guess, r0_guess])
    D_fit, a_fit, r0_fit = params

    print("Morse fit parameters (project form):")
    print(f"  D      = {D_fit:.6f}")
    print(f"  a      = {a_fit:.6f}")
    print(f"  r0     = {r0_fit:.6f}  (bond length)")
    print(f"  E_min  = {morse_total(r0_fit, D_fit, a_fit, r0_fit, E_single):.6f}")

    # Smooth curve for plotting
    r_fit = np.linspace(r_data.min(), r_data.max(), 400)
    E_fit = morse_total(r_fit, D_fit, a_fit, r0_fit, E_single)

    # ---- Plot raw MC data + Morse fit ----
    plt.figure(figsize=(6, 5), clear = True)
    plt.title("")
    plt.scatter(r_data, energies, color="black", label="MC energies")
    plt.plot(r_fit, E_fit, "r--", label="Morse fit")

    plt.xlabel("Bond length r (a.u.)")
    plt.ylabel("Energy (a.u.)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Return the fitted Morse parameters
    return D_fit, a_fit, r0_fit


theta1, theta2, theta3, percentage_theta = monte_carlo_minimisation_2protons(0.8, 0.4, pdf_xyz, 200, 1, q_1, q_2, q, THETA_INITIAL)
print("Optimized theta values for H2 molecule:", theta1, theta2, theta3)
print(f"Percentage of theta accepted: {percentage_theta:.2f}%")

#energy_estimate = energy(metropolis_2protons_multi(0.8, pdf_xyz, 10000, theta1, theta2, theta3, q_1, q_2), theta1, theta2, theta3, q)
#print("Estimated Energy Expectation Value for H2 molecule (10000 walkers):", energy_estimate)

plot_h2_pdf_histogram(theta1, theta2, theta3, q_1, q_2)

#D_fit, a_fit, r0_fit = bond_length(step_size_r=1.1, pdf=pdf_xyz, iterations=100, T=0.5, r_min=0.2, r_max=2.5, THETA_INITIAL=THETA_INITIAL, E_single=E_single)
#print(D_fit, a_fit, r0_fit)

#-------------------------------------------
end = time.time()
print(f"Run time: {end - start:.5f} seconds")
