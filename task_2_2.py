import time
start = time.time()
#------------------

import numpy as np 
import matplotlib.pyplot as plt
from task_2_1 import w2_2nd_general

#plot pdf
r = np.arange(-5, 5, 0.1)

def pdf0(r):
    return np.exp(- r ** 2) / np.sqrt(np.pi)

#plt.plot(r, pdf0(r), label='PDF')
#plt.show()

# Metropolis-Hastings algorithm
def metropolis_hastings(step_size, pdf, iterations):

    # Pre-generate all uniforms we'll need
    # 2 per step (one for proposal, one for accept), plus one for initial x
    u = np.random.rand(iterations * 2 + 1)

    x = np.zeros(iterations)
    # start near 0 instead of at extreme
    x[0] = (u[0] - 0.5) * 2   # initial point in [-1, 1]

    for i in range(1, iterations):
        x_current = x[i-1]

        # proposal step: symmetric uniform in [-step_size, step_size]
        u_step   = u[2*i - 1]
        u_accept = u[2*i]

        x_proposed = x_current + step_size * (2*u_step - 1)

        # Metropolis acceptance ratio
        alpha = min(1.0, pdf(x_proposed) / pdf(x_current))

        if u_accept < alpha:
            x[i] = x_proposed
        else:
            x[i] = x_current

    return x

samples = metropolis_hastings(0.5, pdf0, 1000000)

def metropolis_hastings_multi(step_size, pdf, iterations, n_walkers):
    """
    Run n_walkers independent 1D Metropolis–Hastings chains in parallel.

    Returns
    -------
    samples : 1D ndarray
        Flattened array of shape (iterations * n_walkers,),
        i.e. all walkers' samples concatenated.
    """

    # Total uniforms:
    # 1 per walker for initial point
    # 2 per step per walker (proposal + accept)
    u = np.random.rand(n_walkers * (1 + 2 * (iterations - 1)))
    idx = 0

    # (iterations, n_walkers)
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

        # Avoid division by zero: if p_current == 0, force accept (or handle separately)
        alpha = np.ones(n_walkers)
        valid = p_current > 0
        alpha[valid] = np.minimum(1.0, p_prop[valid] / p_current[valid])

        accept = u_accept < alpha

        accepted_count += accept.sum()
        x[i] = np.where(accept, x_proposed, x_current)

    # Flatten all walkers into one long array
    return x.reshape(iterations * n_walkers), accepted_count / (iterations * n_walkers)

samples_multi, fraction = metropolis_hastings_multi(2, pdf0, 50, 100000)

print("percentage of accepted steps:", fraction * 100)

#plot histogram of samples
plt.hist(samples_multi, bins=50, density=True, alpha=0.6, label='Samples Histogram')
#plot pdf  
r = np.arange(-5, 5, 0.1)
plt.plot(r, pdf0(r), label='PDF', color='red')
plt.title('Metropolis-Hastings Sampling')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()

#----------------------------------------------------------------------------

#compute energy expectation value

def phi0(x):
    return np.exp(- x ** 2 / 2)

energy = 0
#for i in range(100000):
#    energy += 1/100000 * (- 1 / 2 * w2_2nd_general(phi0, samples[i], 0.001) / phi0(samples[i]) + 1/2 * samples[i] ** 2)

#print("Estimated Energy Expectation Value for n = 0:", energy)

def phi1(x):
    return 2 * x * np.exp(- x ** 2 / 2)

#energy = 0
#for i in range(100000):
    energy += 1/100000 * (- 1 / 2 * w2_2nd_general(phi1, samples[i], 0.001) / phi1(samples[i]) + 1/2 * samples[i] ** 2)

#print("Estimated Energy Expectation Value for n = 1:", energy)

#-------------------------------------------
end = time.time()
print(f"Run time: {end - start:.5f} seconds")


