## Option 1: Analytically tractable mixed posterior with discrete–continuous coupling

### Context
A common reason mixed-parameter inference is hard is that *discrete structure and continuous parameters couple* in the likelihood. This creates a posterior that is neither purely continuous nor purely discrete, but a **mixture**: the continuous posterior changes depending on the discrete configuration, and the discrete posterior depends on how well each configuration explains the data.

Option 1 is the minimal analytic model that already exhibits this: a single Bernoulli variable \(z\) *shifts* the mean of a Gaussian observation, and a continuous parameter \(\theta\) further controls that mean. The posterior becomes a **two-component mixture of Gaussians** in \(\theta\), with data-dependent mixing weights \(p(z\mid x)\).

This is still fully tractable (no MCMC needed), so it’s ideal as an opening “ground-truth” mixed example that is more realistic than the fully factorized toy.

---

## Model specification

### Parameters (mixed)
- Continuous: \(\theta \in \mathbb{R}\)
- Discrete: \(z \in \{0,1\}\)

### Prior
\[
\theta \sim \mathcal{N}(0,1), \qquad z \sim \mathrm{Bernoulli}(1/2).
\]

### Likelihood (coupled)
\[
x \mid \theta, z \sim \mathcal{N}(\theta + a z,\ \sigma^2),
\]
where \(a \in \mathbb{R}\) is a known shift magnitude and \(\sigma>0\) is known observation noise.

Interpretation: the discrete variable selects a “regime” (baseline vs shifted by \(a\)), while \(\theta\) provides a continuous adjustment within that regime.

---

## Ground-truth posterior (closed form)

### 1) Conditional continuous posterior \(p(\theta \mid x, z)\)

Define the shifted observation
\[
y_z := x - a z.
\]
Then
\[
y_z \mid \theta \sim \mathcal{N}(\theta,\ \sigma^2),
\]
and with the prior \(\theta\sim \mathcal{N}(0,1)\) we obtain conjugate Gaussian updating:
\[
\theta \mid x, z \sim \mathcal{N}(m_z,\ v),
\]
with
\[
v = \left(1 + \frac{1}{\sigma^2}\right)^{-1} = \frac{\sigma^2}{1+\sigma^2},
\qquad
m_z = v\cdot \frac{y_z}{\sigma^2} = \frac{x-a z}{1+\sigma^2}.
\]

So the conditional posterior is always Gaussian with **fixed variance** \(v\), but its mean shifts depending on \(z\).

---

### 2) Discrete posterior \(p(z \mid x)\)

Integrate out \(\theta\) to obtain the marginal likelihood for each \(z\):
\[
p(x \mid z) = \int p(x\mid \theta,z)\,p(\theta)\,d\theta.
\]
Since both are Gaussian, the marginal is Gaussian:
\[
x \mid z \sim \mathcal{N}(a z,\ 1+\sigma^2).
\]
With the uniform prior over \(z\), the posterior is:
\[
p(z \mid x) = \frac{\mathcal{N}\!\left(x;\ a z,\ 1+\sigma^2\right)}
{\mathcal{N}\!\left(x;\ 0,\ 1+\sigma^2\right)+\mathcal{N}\!\left(x;\ a,\ 1+\sigma^2\right)}.
\]

A useful equivalent “logistic form” for \(p(z=1\mid x)\) is:
\[
p(z=1\mid x) = \sigma\!\left(\frac{a x - \tfrac{1}{2}a^2}{1+\sigma^2}\right),
\quad
\sigma(t)=\frac{1}{1+e^{-t}}.
\]
This makes the behavior transparent:
- larger \(x\) increases posterior mass on \(z=1\),
- larger \(a\) makes the discrete regimes more separable,
- larger \(\sigma\) smooths the transition.

---

### 3) Marginal continuous posterior \(p(\theta \mid x)\)

By the law of total probability over \(z\),
\[
p(\theta \mid x) = \sum_{z\in\{0,1\}} p(\theta \mid x,z)\,p(z\mid x).
\]
Since each conditional is Gaussian, the marginal is a **two-component Gaussian mixture**:
\[
p(\theta \mid x) = p(z=0\mid x)\,\mathcal{N}(\theta;\ m_0,\ v)
\;+\;
p(z=1\mid x)\,\mathcal{N}(\theta;\ m_1,\ v),
\]
where
\[
m_0 = \frac{x}{1+\sigma^2},\qquad
m_1 = \frac{x-a}{1+\sigma^2},\qquad
v = \frac{\sigma^2}{1+\sigma^2}.
\]

This mixture structure is exactly the kind of mixed posterior behavior that causes difficulties for generic likelihood-based samplers on larger problems (multi-modality, discrete–continuous coupling), but here it remains analytically computable.

---

## Why this is a stronger opening example than the factorized toy
- It introduces **true coupling**: \(x\) informs \(z\) and \(\theta\) jointly; the posterior over \(\theta\) depends on \(z\).
- The continuous posterior is not a single Gaussian but a **mixture**, which is the simplest form of multi-modality induced by discrete structure.
- Ground truth remains **exact and cheap**: compute \(p(z\mid x)\) by evaluating two Gaussians; compute \(p(\theta\mid x,z)\) by conjugate update.

This gives you a clean, fully analytical benchmark that already reflects the core challenge MNPE is meant to solve.