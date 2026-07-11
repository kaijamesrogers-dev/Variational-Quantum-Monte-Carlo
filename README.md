# Variational Monte Carlo of the Hydrogen Molecule

[日本語](#日本語)

This project uses a computational method called **Variational Monte Carlo (VMC)**
to predict two things about the hydrogen molecule (H₂):

1. How far apart its two atoms naturally sit (the **bond length**)
2. How much energy is released when the bond forms (the **binding energy**)

Both predictions came out within about 10% of the experimentally measured values.

---

## Why this needs a computational method

A molecule naturally settles into whatever configuration has the least energy,
so predicting a molecule's structure means finding its lowest-energy
configuration. For a single hydrogen atom this can be solved exactly on paper.
But as soon as a system contains two or more electrons, the repulsion between
the electrons makes the governing equation (the Schrödinger equation)
impossible to solve exactly. The hydrogen molecule is the simplest system where
this happens, which makes it an ideal test case: the exact answer can't be
derived, but the real values are well measured, so the accuracy of the method
can be checked.

Brute-force computation doesn't work either. The object being computed is the
**wavefunction** — the function that describes where the electrons are likely
to be found. Storing it on a grid requires memory that grows exponentially with
the number of coordinates: two electrons in 3D means a six-dimensional
function, and even a coarse grid of 100 points per axis would need
100⁶ ≈ a trillion values. This is what motivates methods based on random
sampling instead.

---

## How the method works

Three components, each with one job:

**Monte Carlo integration** estimates the molecule's energy for a given trial
wavefunction. The energy is defined mathematically as an integral over all
possible positions of every electron, but this integral has no closed-form
solution once more than one electron is involved. Monte Carlo integration gets
around this by approximating the integral as an average: it samples a large
number of electron positions, calculates the energy at each one, and averages
the results. The more samples used, the more accurate the estimate.

**The Metropolis algorithm** is the method used to generate those samples
correctly. The samples can't just be picked uniformly at random — they need to
be distributed according to the probability density of the wavefunction, |ψ|²,
so that positions where an electron is more likely to be found are sampled
more often. The Metropolis algorithm achieves this by proposing a small random
step from the current position, then accepting or rejecting it based on the
ratio of probability densities at the new and old positions. A move to a more
probable position is always accepted, while a move to a less probable position
is accepted only with a probability equal to that ratio. Repeating this many
times produces a sequence of positions whose overall distribution converges to
|ψ|² — exactly the distribution needed for the Monte Carlo estimate above to
be valid.

**Simulated annealing** optimises the parameters of the trial wavefunction
itself. A "trial" wavefunction is a parameterised first guess at the true
wavefunction: its exact shape depends on one or more unknown parameters, and
the values that minimise the energy must be found by search. A useful
guarantee called the **variational principle** makes this search well-defined:
the energy estimated from any trial wavefunction can never fall below the true
minimum, so "lower is always better" is a safe rule. Simulated annealing
searches by proposing a random change to a parameter, accepting it immediately
if it lowers the energy, and accepting it with a probability that decreases
over time if it raises the energy. This controlled chance of accepting a worse
move lets the search escape local minima rather than converging prematurely on
the first reasonable value it finds. As the search progresses, that acceptance
probability is steadily reduced (referred to as "cooling"), and the search
settles on a final set of parameter values.

Together, these three methods form a loop: a candidate set of wavefunction
parameters is proposed by simulated annealing, the Metropolis algorithm
generates electron position samples consistent with that wavefunction, and
Monte Carlo integration turns those samples into an energy estimate, which is
fed back to simulated annealing to decide whether to accept the proposed
parameters. The lowest-energy result of this loop is the project's prediction
for the configuration of the hydrogen molecule.

---

## File 1: `finite_difference_test.py` — validating the numerical derivative

The energy calculations depend on computing a second derivative of the
wavefunction at sampled points. Computers can't perform exact calculus on an
arbitrary function, so this is approximated instead, using a formula known as
the **2nd-order central-difference scheme**. This file checks that the
approximation is accurate before it is used anywhere else.

The test uses the ground-state wavefunction of the harmonic oscillator, a
standard system whose exact second derivative is known analytically. The
numerical approximation is compared against this exact value across a wide
range of step sizes, to find where the approximation is reliable. The 4th-order
central-difference formula is also compared for reference.

**Results:**

![Finite-difference error scaling](Figures/Figure_1.png)

For larger step sizes, the error decreases as the step size shrinks, exactly
as predicted by the formula's theoretical accuracy. Below a certain step size,
however, the error starts increasing again. This happens because the
calculation involves subtracting nearly-equal numbers, and a computer can only
store numbers to a fixed number of significant digits — once the step size is
small enough, that subtraction is dominated by floating-point rounding error
rather than the true difference being measured. This file identifies the step
size that minimises total error, and that step size is used in every later
file.

---

## File 2: `metropolis_1d_test.py` — validating the Metropolis sampler in 1D

This file tests the Metropolis algorithm described above on the simplest
possible system: a single particle in one dimension, under the harmonic
oscillator potential. The exact ground-state and first-excited-state energies
for this system are known from theory, so this is a controlled test of whether
the sampling method itself is implemented correctly, before extending it to
the much harder 3D and 6D cases that follow.

The algorithm generates a large number of sample positions, and the resulting
distribution is compared against the known analytical probability density. The
same samples are then used to compute a Monte Carlo estimate of the energy for
both the ground state and the first excited state.

**Results:**

![Metropolis histogram vs analytical PDF](Figures/Figure_2.png)

*The histogram of sampled positions (blue) matches the exact probability
density (red), confirming the Metropolis algorithm is sampling correctly.*

The energy estimates came out as 0.4999997 for the ground state and 1.4999955
for the first excited state, against exact values of 0.5 and 1.5 — agreement
to within 10⁻⁶.

---

## File 3: `vmc_hydrogen_atom_3d.py` — the hydrogen atom in 3D, with an unknown parameter

This file extends the method to a single hydrogen atom (one proton, one
electron) in full three-dimensional space. The exact ground-state energy for
this system is also known, so it still serves as a validation case — but it
introduces the key new ingredient: a parameter whose value has to be found by
search.

The trial wavefunction used here, ψ(r; θ) = e^(−θr), depends on a single
variational parameter, θ. Simulated annealing searches over θ exactly as
described above: at each step a new θ is proposed, its energy is estimated via
Metropolis sampling, and the move is accepted or rejected according to the
annealing rule.

**Results:**

![Theta and energy convergence](Figures/Figure_3.png)

*θ (blue) and energy (red) both converge as the simulated annealing search
progresses.*

The search settled at θ = 0.987 with an energy estimate of E = −0.49998,
within 10⁻⁴ of the exact values (θ = 1, E = −0.5).

---

## File 4: `vmc_hydrogen_molecule.py` — the hydrogen molecule

This file applies the full method to the actual system of interest: the
hydrogen molecule, H₂ (two protons and two electrons). This is the case where
the governing equation has no exact solution, so the method built up in
Files 1–3 is now doing real work.

The trial wavefunction used here is of a standard form known as the
**Slater–Jastrow** form. You don't need to know its details to keep reading —
the two things that matter are that (a) it allows either electron to sit near
either proton, and (b) it includes a factor that explicitly accounts for the
two electrons repelling each other. It depends on three variational parameters
(θ₁, θ₂, θ₃), rather than the single parameter used for the atom, and all
three are optimised jointly by simulated annealing.

Repeating this optimisation across a range of proton separations gives an
energy for each bond length. Fitting the resulting curve to a **Morse
potential** — a standard model for how a bond's energy varies with its
length — gives the project's final predictions:

- **Bond length:** 1.3223 a.u., about 5.6% below the measured value of
  1.40 a.u.
- **Minimum energy:** −1.1565 a.u. (statistical uncertainty ±0.001 a.u.), with
  the dissociation energy about 8% below the measured value.

The statistical uncertainty is roughly ten times smaller than the gap to the
measured values, which shows the remaining discrepancy comes from the limited
flexibility of the chosen wavefunction form (the restricted ansatz), not from
sampling noise.

**Results:**

![Electron density map](Figures/Figure_4.png)

*Each electron's sampled positions are plotted as a 2D probability density.
The concentration of density between the two protons (marked X) is consistent
with a bonding molecular orbital.*

---

## Tools used

Python, NumPy (numerical arrays and vectorised computation), Matplotlib
(plotting), SciPy (curve fitting for the Morse potential).

---

## Report

The full write-up (method, results, error analysis) is in
[`report.pdf`](VMC_Report.pdf).

---

# 日本語

[English](#variational-monte-carlo-of-the-hydrogen-molecule)

# 水素分子の変分モンテカルロ法

本プロジェクトでは、**変分モンテカルロ法（VMC）**という計算手法を用いて、水素分子（H₂）について次の2つを予測する:

1. 2つの原子が自然に落ち着く距離（**結合長**）
2. 結合が形成される際に放出されるエネルギー（**結合エネルギー**）

いずれの予測値も、実験による実測値との差は約10%以内に収まった。

---

## なぜ計算手法が必要か

分子は最もエネルギーの低い配置に自然と落ち着く。したがって分子の構造を予測することは、最小エネルギーの配置を探すことに等しい。水素原子1個であれば厳密に解けるが、電子が2個以上になると、電子同士の反発によって支配方程式（シュレーディンガー方程式）は厳密には解けなくなる。水素分子はこの問題が生じる最も単純な系であり、厳密解は存在しない一方で実測値はよく知られているため、手法の精度を検証するのに理想的な対象である。

総当たり的な計算も現実的ではない。計算の対象となるのは**波動関数**（電子がどこに見つかりやすいかを表す関数）だが、これを格子上に保存するには座標の数に対して指数関数的に増えるメモリが必要になる。電子2個の3次元系は6次元の関数となり、各軸100点という粗い格子でも100⁶（約1兆）個の値が必要となる。ここから、ランダムサンプリングに基づく手法の必要性が生じる。

---

## 手法の仕組み

3つの要素が、それぞれ1つの役割を担う。

**モンテカルロ積分**は、与えられた試行波動関数に対するエネルギーの推定に用いる。エネルギーは全電子のあらゆる位置にわたる積分として定義されるが、電子が複数になるとこの積分は解析的に解けない。モンテカルロ積分では、積分を平均で近似することでこの問題を回避する: 多数の電子位置をサンプリングし、それぞれの位置でエネルギーを計算し、その平均を取る。サンプル数が増えるほど推定精度は上がる。

**メトロポリス法**は、そのサンプルを正しく生成するためのアルゴリズムである。サンプルは一様なランダムでは不十分で、波動関数の確率密度|ψ|²に従って分布させる必要がある（電子が見つかりやすい位置ほど頻繁にサンプリングされるように）。メトロポリス法では、現在位置から小さなランダムステップを提案し、新旧位置の確率密度の比に基づいて採否を決める。より確率の高い位置への移動は必ず採用し、より低い位置への移動はその比に等しい確率で採用する。これを多数回繰り返すと、サンプル全体の分布が|ψ|²に収束し、上記のモンテカルロ推定が成り立つための条件が満たされる。

**Simulated annealing（焼きなまし法）**は、試行波動関数そのもののパラメータ最適化に用いる。「試行」波動関数とは、真の波動関数に対するパラメータ付きの初期推測のことであり、その正確な形は未知のパラメータに依存する。エネルギーを最小化するパラメータの値は事前には分からないため、探索して求める必要がある。ここで役立つのが**変分原理**という保証で、どんな試行波動関数から推定したエネルギーも真の最小値を下回ることはない。つまり探索において「低いほど良い」が常に成り立つ。Simulated annealingでは、パラメータへのランダムな変更を提案し、エネルギーが下がれば即座に採用し、上がる場合も時間とともに減少する確率で採用する。この「悪化を許容する確率」により、最初に見つかったそれらしい値に早々に収束するのではなく、局所最小値から抜け出すことができる。探索が進むにつれてこの受理確率を少しずつ下げていき（「冷却」と呼ぶ）、最終的なパラメータ値に収束させる。

3つの要素はループを構成する: simulated annealingが波動関数のパラメータ候補を提案し、メトロポリス法がその波動関数に従う電子位置サンプルを生成し、モンテカルロ積分がサンプルからエネルギーを推定し、その結果がsimulated annealingにフィードバックされて採否が決まる。このループで得られる最小エネルギー配置が、本プロジェクトの水素分子に対する予測となる。

---

## ファイル1：`finite_difference_test.py` — 数値微分の検証

エネルギー計算には、サンプル点での波動関数の2階微分が必要になる。コンピュータは任意の関数を厳密に微分できないため、**2次中心差分**という公式で近似する。本ファイルでは、この近似を他の計算で使う前に精度を検証する。

テスト対象は調和振動子の基底状態波動関数で、2階微分の厳密解が解析的に知られている標準的な系である。数値近似を厳密値と広範なステップ幅にわたって比較し、近似が信頼できる範囲を特定する。参考として4次中心差分公式との比較も行う。

**結果：**

![有限差分誤差のスケーリング](Figures/Figure_1.png)

大きなステップ幅では、理論予測通りステップ幅の縮小とともに誤差が減少する。しかし一定以下のステップ幅では誤差が増加に転じる。差分計算はほぼ等しい数同士の引き算を含み、コンピュータは有限の桁数でしか数を保存できないため、ステップ幅が小さすぎると測りたい真の差ではなく浮動小数点の丸め誤差が支配的になるからである。本ファイルで全体の誤差を最小にするステップ幅を特定し、以降のすべてのファイルで使用する。

---

## ファイル2：`metropolis_1d_test.py` — 1次元でのメトロポリス法の検証

上で説明したメトロポリス法を、最も単純な系（1次元調和振動子ポテンシャル下の1粒子）でテストする。この系の基底状態および第1励起状態のエネルギーは理論的に既知であるため、後に続くより難しい3次元・6次元の問題に拡張する前の、サンプリング手法そのものの実装検証として機能する。

アルゴリズムにより大量のサンプル位置を生成し、その分布を既知の解析的確率密度と比較する。同じサンプルを用いて、基底状態および第1励起状態のモンテカルロエネルギー推定も行う。

**結果：**

![メトロポリス法のヒストグラムと解析的PDF](Figures/Figure_2.png)

*サンプル位置のヒストグラム（青）が厳密な確率密度（赤）と一致しており、メトロポリス法が正しくサンプリングできていることが確認できる。*

エネルギー推定値は基底状態で0.4999997、第1励起状態で1.4999955となり、厳密値の0.5および1.5と10⁻⁶以内で一致した。

---

## ファイル3：`vmc_hydrogen_atom_3d.py` — 3次元水素原子と未知パラメータ

手法を水素原子1個（陽子1個・電子1個）の完全な3次元空間に拡張する。この系の基底状態エネルギーも既知であるため引き続き検証用として機能するが、ここで新しい要素が加わる: 探索によって値を求めなければならないパラメータである。

使用する試行波動関数ψ(r; θ) = e^(−θr)は、1つの変分パラメータθに依存する。上で説明した通りsimulated annealingでθを探索する: 各ステップで新しいθを提案し、メトロポリス法によるサンプリングでそのエネルギーを推定し、annealingの規則に従って採否を決める。

**結果：**

![θとエネルギーの収束](Figures/Figure_3.png)

*探索が進むにつれてθ（青）とエネルギー（赤）がともに収束する。*

探索はθ = 0.987、エネルギー推定値E = −0.49998で収束し、厳密値（θ = 1、E = −0.5）と10⁻⁴以内で一致した。

---

## ファイル4：`vmc_hydrogen_molecule.py` — 水素分子

本来の対象である水素分子H₂（陽子2個・電子2個）に完全な手法を適用する。支配方程式に厳密解が存在しないのはこのケースであり、ファイル1〜3で構築・検証した手法がここで本来の仕事をする。

ここで使う試行波動関数は、**スレーター–ジャストロー型**と呼ばれる標準的な形の関数である。詳細を知らなくても読み進められるが、重要な点は2つある: (a) どちらの電子もどちらの陽子の近くにも存在できる形になっていること、(b) 電子同士の反発を明示的に取り入れる因子を含むこと。水素原子で使った1パラメータに対し、3つの変分パラメータ（θ₁, θ₂, θ₃）を持ち、3つを同時にsimulated annealingで最適化する。

この最適化を陽子間距離を変えながら繰り返すと、結合長ごとのエネルギーが得られる。この曲線を**モースポテンシャル**（結合のエネルギーが結合長とともにどう変化するかを表す標準的なモデル）にフィットして、最終的な予測値を得た:

- **結合長:** 1.3223 a.u. — 実測値1.40 a.u.より約5.6%小さい
- **最小エネルギー:** −1.1565 a.u.（統計誤差 ±0.001 a.u.）。解離エネルギーは実測値より約8%小さい

統計誤差は実測値とのずれの10分の1程度であり、残る差はサンプリングのノイズではなく、選んだ波動関数の形の制約（restricted ansatz）に起因することが分かる。

**結果：**

![電子密度マップ](Figures/Figure_4.png)

*各電子のサンプル位置を2次元確率密度としてプロットしたもの。2つの陽子（×印）の間に密度が集中しており、結合性分子軌道と整合する。*

---

## 使用ツール

Python、NumPy（数値配列・ベクトル化演算）、Matplotlib（グラフ描画）、SciPy（モースポテンシャルのカーブフィッティング）。

---

## レポート

詳細な手法・結果・誤差解析は [`report.pdf`](VMC_Report.pdf) を参照。
