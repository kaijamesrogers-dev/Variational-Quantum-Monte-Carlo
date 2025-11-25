import numpy as np
import matplotlib.pyplot as plt

def w(x):
    # wave function for n = 0
    return np.exp(-x**2/2)

def w2_exact(x):
    # analytic second derivative of wave function
    return (x**2 - 1) * np.exp(-x**2/2)

def w2_2nd(x, h):
    # 3-point (2nd order) central difference
    return (w(x+h) - 2*w(x) + w(x-h)) / h**2

def w2_4th(x, h):
    # 5-point (4th order) central difference
    return (16*w(x-h) + 16*w(x+h) -w(x+2*h) - w(x-2*h) - 30*w(x)) / (12*h**2)

# Range of h values to test
hs = np.linspace(0, 1, 100)[1:]  # avoid h=0
x0 = 0.3   # choose any point you like

errors_2nd = []
errors_4th = []

for h in hs:
    errors_2nd.append(abs(w2_2nd(x0,h) - w2_exact(x0)))
    errors_4th.append(abs(w2_4th(x0,h) - w2_exact(x0)))

# Plot
plt.plot(hs, errors_2nd, label='3-point (2nd order)')
plt.plot(hs, hs**2, '--', label='h^2', color='gray')
plt.title('Log Absolute Error in Second Derivative Approximation')
plt.xlabel('Log(h)')
plt.ylabel('Log Absolute Error')
plt.grid(True)
plt.legend()
plt.show()

