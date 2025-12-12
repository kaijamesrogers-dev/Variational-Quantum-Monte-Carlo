import time
start = time.time()
#------------------

import numpy as np 
import matplotlib.pyplot as plt

#Match text to lab report
plt.rcParams.update({
    'font.size': 14, 
    'axes.titlesize': 14,  
    'axes.labelsize': 14, 
    'ytick.labelsize': 14,
    'legend.fontsize': 14})

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False})


# General 2nd order central difference function
def w2_2nd_general(f, x, h, *f_args):
    return (f(x + h, *f_args) - 2*f(x, *f_args) + f(x - h, *f_args)) / h**2

#plot pdf
r = np.arange(-5, 5, 0.1)

def pdf0(r):
    return np.exp(- r ** 2) / np.sqrt(np.pi)

#plt.plot(r, pdf0(r), label='PDF')
#plt.show()

# Metropolis algorithm
def metropolis(step_size, pdf, iterations):

    u = np.random.rand(iterations * 2 + 1)
    accepted_count = 0

    x = np.zeros(iterations)
    x[0] = (u[0] - 0.5) * 2   # initial point in [-1, 1]

    for i in range(1, iterations):
        x_current = x[i-1]

        # proposal step
        u_step   = u[2*i - 1]
        u_accept = u[2*i]
        x_proposed = x_current + step_size * (2*u_step - 1)

        # Metropolis acceptance ratio
        alpha = min(1.0, pdf(x_proposed) / pdf(x_current))
        accept = u_accept < alpha
        accepted_count += accept.sum()
        x[i] = np.where(accept, x_proposed, x_current)

    return x, accepted_count / iterations * 100

N = 1000000
samples, percentage = metropolis(2, pdf0, N)

def metropolis_multi(step_size, pdf, iterations, n_walkers):
    """
    Run n_walkers independent 1D Metropolis chains in parallel.

    Returns
    -------
    samples : 1D ndarray
        Flattened array of shape (iterations * n_walkers,),
    """
    # pre-generate all random numbers
    u = np.random.rand(n_walkers * (1 + 2 * (iterations - 1)))
    idx = 0

    x = np.zeros((iterations, n_walkers))

    # Initial positions in [-1, 1]
    x[0] = (u[idx:idx + n_walkers] - 0.5) * 2.0
    idx += n_walkers

    accepted_count = 0

    for i in range(1, iterations):
        x_current = x[i - 1]

        # Proposal uniforms and accept uniforms for all walkers
        u_step   = u[idx:idx + n_walkers]; idx += n_walkers
        u_accept = u[idx:idx + n_walkers]; idx += n_walkers

        # Propose new positions
        x_proposed = x_current + step_size * (2 * u_step - 1)

        # Vectorised pdf evaluation
        p_current = pdf(x_current)
        p_prop    = pdf(x_proposed)

        # Avoid division by zero
        alpha = np.ones(n_walkers)
        valid = p_current > 0
        alpha[valid] = np.minimum(1.0, p_prop[valid] / p_current[valid])

        accept = u_accept < alpha
        accepted_count += accept.sum()
        x[i] = np.where(accept, x_proposed, x_current)

    return x.reshape(iterations * n_walkers), accepted_count / (iterations * n_walkers)

#samples_multi, fraction = metropolis_hastings_multi(2, pdf0, 100, 100000)

#print("percentage of accepted steps:", percentage)

#plot histogram of samples
plt.hist(samples, bins=30, density=True, alpha=0.6, label='Samples')
#plot pdf  
r = np.arange(-5, 5, 0.1)
plt.plot(r, pdf0(r), label='PDF', color='red')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

#----------------------------------------------------------------------------

#compute energy expectation value

def phi0(x):
    return np.exp(- x ** 2 / 2)

energy = 0
for i in range(N):
    energy += 1/N * (- 1 / 2 * w2_2nd_general(phi0, samples[i], 0.0002) / phi0(samples[i]) + 1/2 * samples[i] ** 2)

print("Estimated Energy Expectation Value for n = 0:", energy)

def phi1(x):
    return 2 * x * np.exp(- x ** 2 / 2)

energy = 0
for i in range(N):
    energy += 1/N * (- 1 / 2 * w2_2nd_general(phi1, samples[i], 0.0002) / phi1(samples[i]) + 1/2 * samples[i] ** 2)

print("Estimated Energy Expectation Value for n = 1:", energy)

#-------------------------------------------
end = time.time()
#print(f"Run time: {end - start:.5f} seconds")

plt.show()
