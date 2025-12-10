import time
start = time.time()

import numpy as np
import matplotlib.pyplot as plt

def w2_2nd_general(f, x, h, *f_args):
    return (f(x + h, *f_args) - 2*f(x, *f_args) + f(x - h, *f_args)) / h**2

theta = 1.0

def psix(x, xyz, theta):
    return np.exp(- theta * np.sqrt(x ** 2 + xyz[1] ** 2 + xyz[2] ** 2))

def psiy(x, xyz, theta):
    return np.exp(- theta * np.sqrt(xyz[0] ** 2 + x ** 2 + xyz[2] ** 2))

def psiz(x, xyz, theta):
    return np.exp(- theta * np.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + x ** 2))

def pdf_xyz(x, y, z, theta):
    r = np.sqrt(x**2 + y**2 + z**2)
    return (theta**3 / np.pi) * np.exp(-2 * theta * r)

def metropolis_hastings_3d(step_size, pdf, iterations, theta):
    """
    3D Metropolis–Hastings sampler.

    Returns:
        samples: array of shape (iterations, 3)
                 each row is [x, y, z]
    """
    # 3 uniforms for initial point + (3 for proposal + 1 for accept) per step
    n_uniform = 3 + 4 * (iterations - 1)
    u = np.random.rand(n_uniform)
    idx = 0
    accepted_number = 0

    samples = np.zeros((iterations, 3))

    # Initial point in [-1, 1]^3
    samples[0, :] = (u[idx:idx+3] - 0.5) * 2.0
    idx += 3

    for i in range(1, iterations):
        x_current, y_current, z_current = samples[i-1, :]

        # Proposal step in each coordinate
        u_step = u[idx:idx+3]
        idx += 3
        step_vec = step_size * (2 * u_step - 1)  # each in [-step_size, step_size]

        x_prop = x_current + step_vec[0]
        y_prop = y_current + step_vec[1]
        z_prop = z_current + step_vec[2]

        # Acceptance uniform
        u_accept = u[idx]
        idx += 1

        p_current = pdf(x_current, y_current, z_current, theta)
        p_prop    = pdf(x_prop,    y_prop,    z_prop,    theta)

        if p_current <= 0:
            alpha = 1.0    # if current state is impossible, move
        else:
            alpha = min(1.0, p_prop / p_current)

        if u_accept < alpha:
            samples[i, :] = [x_prop, y_prop, z_prop]
            accepted_number += 1
        else:
            samples[i, :] = [x_current, y_current, z_current]

    return samples, accepted_number / iterations * 100

#samples = metropolis_hastings_3d(0.5, pdf_xyz, 100000, theta, SEED)

def energy(samples, theta):
    r = np.linalg.norm(samples, axis=1)
    N = samples.shape[0]
    E = 0.0
    h = 2e-4  # slightly bigger h is fine

    for i in range(N):
        xyz = samples[i]
        ri  = r[i]
        if ri == 0:
            continue

        lap = (w2_2nd_general(psix, xyz[0], h, xyz, theta) +
               w2_2nd_general(psiy, xyz[1], h, xyz, theta) +
               w2_2nd_general(psiz, xyz[2], h, xyz, theta))

        psi_val = np.exp(-theta * ri)

        El = -0.5 * lap / psi_val - 1.0 / ri
        E += El

    return E / N

#print("Estimated Energy Expectation Value:", energy(samples, theta))

def monte_carlo_minimisation(step_size, pdf, iterations, T, theta = theta):
    u = np.random.rand(2 * iterations)
    n = 0
    samples, percentage = metropolis_hastings_3d(1.1, pdf, iterations * 30, theta)
    print(percentage)
    E = energy(samples, theta)
    accepted_number_mc = 0
    for i in range(iterations):
        theta_dash = theta + step_size * (2 * u[i] - 1)
        samples_dash, percentage_dash = metropolis_hastings_3d(1.1, pdf, iterations * 30, theta_dash)
        print(percentage_dash)
        E_dash = energy(samples_dash, theta_dash)

        delta_E = E_dash - E

        if delta_E > 0:
            alpha = np.exp(- delta_E / T)
            random = u[2 * i]
            if random < alpha:
                theta = theta_dash
                E = E_dash
                accepted_number_mc += 1
        else:
            theta = theta_dash
            E = E_dash
            accepted_number_mc += 1

        n += 1
        if n == 10:
            T = T * 0.8
            n = 0

    return theta, accepted_number_mc / iterations * 100

opt_theta, percentage_mc = monte_carlo_minimisation(0.7, pdf_xyz, 100, 0.1, theta)
print("Optimized Theta:", opt_theta)
print("Percentage Monte Carlo:", percentage_mc)

samples_optimized, percentage = metropolis_hastings_3d(1.1, pdf_xyz, 10000, opt_theta)
print("Estimated Energy Expectation Value with Optimized Theta:", energy(samples_optimized, opt_theta))
print(percentage)

end = time.time()
print(f"Run time: {end - start:.5f} seconds")