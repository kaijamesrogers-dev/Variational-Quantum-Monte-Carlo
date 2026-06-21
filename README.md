# Variational Monte Carlo of the Hydrogen Molecule

This project uses a computational method called **Variational Monte Carlo (VMC)**
to predict two things about the hydrogen molecule (H₂):

1. How far apart its two atoms naturally sit (the **bond length**)
2. How much energy is "locked into" that arrangement (the **binding energy**)

---

## Overview

The problem: the configuration of molecules more complex than a single
hydrogen atom cannot be calculated directly, and therefore has to be solved
using computational methods. A molecule naturally settles into whatever
configuration has the least energy. This project aims to find the
configuration of the hydrogen molecule with the least energy, using the
following computational methods.

**Monte Carlo integration** is used to estimate the molecule's energy for a
given trial wavefunction. The energy is defined mathematically as an integral
over all possible positions of every electron — but this integral has no
closed-form solution once more than one electron is involved, and can't be
evaluated on a grid either, since the number of grid points needed grows
exponentially with the number of dimensions. Monte Carlo integration gets
around this by approximating the integral as an average: instead of
integrating over every possible position, it samples a large number of
positions, calculates the energy at each one, and averages the results. The
more samples used, the more accurate the estimate.

**The Metropolis algorithm** is the method used to generate those samples
correctly. The samples can't just be picked uniformly at random — they need
to be distributed according to the probability density of the wavefunction,
|ψ|², so that positions where the electron is more likely to be found are
sampled more often. The Metropolis algorithm achieves this by proposing a
small random step from the current position, then accepting or rejecting it
based on the ratio of probability densities at the new and old positions: a
move to a more probable position is always accepted, while a move to a less
probable position is accepted only with a probability equal to that ratio.
Repeating this many times produces a sequence of sample positions whose
overall distribution converges to |ψ|², which is exactly the distribution
needed for the Monte Carlo estimate above to be valid.

**Simulated annealing** is used to optimise the parameters of the trial
wavefunction itself. The trial wavefunction's exact shape depends on one or
more unknown parameters, and the values of these parameters that minimise
the energy are not known in advance — they must be found by search.
Simulated annealing performs this search by proposing a random change to a
parameter, accepting it immediately if it lowers the energy, and accepting
it with a probability that decreases over time if it raises the energy. This
controlled chance of accepting a worse move allows the search to escape
local minima — parameter values that look optimal in their immediate
neighbourhood but aren't the true minimum — rather than converging
prematurely on the first reasonable value it finds. As the search
progresses, that acceptance probability is steadily reduced (referred to as
"cooling"), so the search becomes increasingly selective and settles on a
final set of parameter values.

Together, these three methods form a loop: a candidate set of wavefunction
parameters is proposed by simulated annealing, the Metropolis algorithm
generates electron position samples consistent with that wavefunction, and
Monte Carlo integration turns those samples into an energy estimate, which
is fed back to simulated annealing to decide whether to accept the proposed
parameters. The final, lowest-energy result of this loop is the project's
prediction for the configuration of the hydrogen molecule.

---

## File 1: `finite_difference_test.py` — validating the numerical derivative

The energy calculations in every later file depend on computing a second
derivative of the wavefunction at sampled points. Computers can't perform
exact calculus on an arbitrary function, so this is approximated instead,
using a method called the 2nd-order central-difference formula. This file
checks that approximation is accurate before relying on it elsewhere.

The test uses the ground-state wavefunction of the harmonic oscillator, a
standard system whose exact second derivative is known analytically. The
numerical approximation is compared against this exact value across a wide
range of step sizes, to find where the approximation is reliable. It also
compares the 4th-order central-difference formula for reference.

**Results:**

![Finite-difference error scaling](images/fig1_finite_difference_error.png)

For larger step sizes, the error decreases as the step size shrinks, exactly
as predicted by the formula's theoretical accuracy. Below a certain step
size, however, the error starts increasing again. This happens because the
calculation involves subtracting nearly-equal numbers, and a computer can
only store numbers to a fixed number of significant digits — once the step
size is small enough, that subtraction is dominated by floating-point
rounding error rather than the true difference being measured. This file
identifies the step size that minimises total error, balancing these two
competing effects, and that step size is used in every later file.

---

## File 2: `metropolis_1d_test.py` — validating the Metropolis sampler in 1D

This file tests the Metropolis algorithm described above, on the simplest
possible system: a single particle in one dimension, under the harmonic
oscillator potential. The exact ground-state and first-excited-state
energies for this system are known from theory, so this is a controlled
test of whether the sampling method itself is implemented correctly, before
extending it to the much harder 3D and 6D cases that follow.

The algorithm generates a large number of sample positions, and the
resulting distribution is compared against the known analytical probability
density. The same samples are then used to compute a Monte Carlo estimate
of the energy, via the local energy formula, for both the ground state and
the first excited state.

**Results:**

![Metropolis histogram vs analytical PDF](images/fig2_metropolis_histogram.png)

The sampled distribution matched the analytical probability density
closely, and the estimated energies agreed with the exact values (0.5 and
1.5, in the units used) to within about one part in a million — confirming
the sampling and energy-estimation procedure is correct before moving to
higher dimensions.

---

## File 3: `vmc_hydrogen_atom_3d.py` — the hydrogen atom in 3D, with an unknown parameter

This file extends the method to a single hydrogen atom (one proton, one
electron) in full three-dimensional space. The exact ground-state energy
for this system is also known, so it still serves as a validation case —
but it introduces a feature that all later, harder problems share.

The trial wavefunction used here, ψ(r; θ) = e^(−θr), depends on a
**variational parameter**, θ, whose correct value isn't known in advance.
Simulated annealing is used to search over θ: at each step, a new θ is
proposed and its energy is estimated; the move is accepted if the energy
decreases, and accepted anyway with a probability that depends on how much
worse it is, and that gradually decreases over the course of the search.
This lets the search avoid settling on a value of θ that looks good locally
but isn't the true minimum.

**Results:**

![Theta and energy convergence](images/fig3_theta_convergence.png)

The search converged to θ ≈ 0.987, very close to the known exact value of
1, and the resulting energy matched the known exact ground-state energy to
within about one part in ten thousand.

---

## File 4: `vmc_h2.py` — the hydrogen molecule

This file applies the full method to the actual system of interest: the
hydrogen molecule, H₂ — two protons and two electrons, with no exact known
solution to validate against, since the underlying equation has no
closed-form answer once electron-electron interaction is included.

The trial wavefunction form is known as the **Slater-Jastrow** form. This
wavefunction depends on three variational parameters (θ₁, θ₂, θ₃), rather
than the single parameter used for the hydrogen atom.

This file:

- Samples electron positions in the resulting six-dimensional space (three
  coordinates per electron) using the Metropolis algorithm
- Estimates the energy for a given set of parameters using the Monte Carlo
  local-energy estimator, now including five Coulomb interaction terms
  (electron-proton attraction for each electron-proton pair, plus
  electron-electron repulsion)
- Uses simulated annealing to search over all three parameters jointly, in
  the same way as File 3
- Repeats this entire optimisation at a range of different separations
  between the two protons, to build up a curve of energy as a function of
  bond length
- Fits that curve to a **Morse potential**, a standard empirical formula for
  the energy of a chemical bond as a function of separation, to extract the
  predicted equilibrium bond length and binding energy

Statistical uncertainty on the final energy estimate is calculated using
block averaging, which accounts for the fact that consecutive Metropolis
samples are correlated rather than fully independent.

**Results:**

![Energy vs bond length with Morse fit](images/fig4_morse_fit.png)

![Electron density map](images/fig5_electron_density.png)

A predicted equilibrium bond length of r₀ = 1.3223 a.u. and minimum energy
of E_min = −1.1565 a.u., compared to experimental values of 1.40 a.u. and a
dissociation energy of 0.17 a.u. — differences of about 5.6% and 8%
respectively. The dominant source of this discrepancy is the restricted
form of the trial wavefunction, rather than Monte Carlo statistical noise,
which is an order of magnitude smaller than the deviation from the
experimental value.

---

## Tools used

Python, NumPy (numerical arrays and vectorised computation), Matplotlib
(plotting), SciPy (curve fitting for the Morse potential).
