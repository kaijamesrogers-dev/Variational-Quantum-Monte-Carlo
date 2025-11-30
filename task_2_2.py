import numpy as np 
import matplotlib.pyplot as plt
from task_2_1 import w2_2nd_general

#plot pdf
r = np.arange(-5, 5, 0.1)

def pdf0(r):
    return np.exp(- r ** 2) / np.sqrt(np.pi)

#plt.plot(r, pdf0(r), label='PDF')
#plt.show()
#----------------------------------------------------------------------------
#generate random numbers
SEED = 7

def lcg_random(SEED, n): #generate n random numbers using LCG
    random_numbers = []
    x = SEED
    for _ in range(n):
        x = (16807 * x) % 2_147_483_647
        random_numbers.append(x / 2_147_483_647)
    return np.array(random_numbers)

random_numbers = lcg_random(SEED, 100000)
#plot histogram of random numbers
#plt.hist(random_numbers, bins=20, density=True)
#plt.title('Histogram of LCG Random Numbers')
#plt.xlabel('Value')
#plt.ylabel('Density')
#plt.show()
#------------------------------------------------------------------------------------
# Metropolis-Hastings algorithm
def metropolis_hastings(step_size, pdf, iterations):

    # Pre-generate all uniforms we'll need
    # 2 per step (one for proposal, one for accept), plus one for initial x
    u = lcg_random(SEED, iterations * 2 + 1)

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

samples = metropolis_hastings(0.5, pdf0, 100000)

#plot histogram of samples
plt.hist(samples, bins=50, density=True, alpha=0.6, label='Samples Histogram')
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
for i in range(100000):
    energy += 1/100000 * (- 1 / 2 * w2_2nd_general(phi0, samples[i], 0.001) / phi0(samples[i]) + 1/2 * samples[i] ** 2)

print("Estimated Energy Expectation Value for n = 0:", energy)

def phi1(x):
    return 2 * x * np.exp(- x ** 2 / 2)

energy = 0
for i in range(100000):
    energy += 1/100000 * (- 1 / 2 * w2_2nd_general(phi1, samples[i], 0.001) / phi1(samples[i]) + 1/2 * samples[i] ** 2)

print("Estimated Energy Expectation Value for n = 1:", energy)


