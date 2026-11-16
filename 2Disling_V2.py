"""2D Ising model: Metropolis sampler (Part A) and Bayesian inference (Part B).

Model
-----
N x N grid of spins s in {-1, +1}, periodic boundaries (N >= 3), energy

    E(s) = -J * B(s),   B(s) = sum over the 2*N**2 nearest-neighbour bonds of
                               s_i * s_j (each bond counted once),

with k_B = 1, so P(s | J, T) = exp(-E(s) / T) / Z = exp((J/T) * B(s)) / Z.

Part A  (spin-level)
    * ``energy``            E(s)
    * ``delta_E``           exact single-flip energy change 2*J*s_i*sum(nbrs)
    * ``metropolis_sweep``  N**2 single-site Metropolis attempts
    * ``run_part_a_tests``  checks delta_E against E(after) - E(before) and
                            checks the Metropolis stationary distribution
                            against the exactly enumerated Boltzmann law.

Part B  (parameter-level)
    * data are drawn by EXACT sampling from the enumerated 4x4 distribution
      (the Metropolis code from Part A is deliberately not used),
    * log L(J, T) = (J/T) * sum_k B(s_k) - K * log Z(J/T),
    * uniform priors J in [0.1, 3], T in [0.5, 6],
    * random-walk Metropolis-Hastings (step size tuned in burn-in only, then
      frozen) is compared with emcee and with a brute-force grid.

Note: the likelihood depends on (J, T) only through r = J/T, so the posterior
over (J, T) is a ridge along lines of constant J/T whose extent is set by the
prior box. Only r is identified by the data; J and T separately are not.

Run as a script::

    python ising_inference_v2.py
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

# ---------------------------------------------------------------------------
# Part A: spin-level model and Metropolis sampler
# ---------------------------------------------------------------------------


def bond_sum(spins):
    """Sum of ``s_i * s_j`` over all nearest-neighbour bonds, each counted once.

    Parameters
    ----------
    spins : ndarray, shape (..., N, N)
        Lattices with entries in {-1, +1}; periodic boundaries; N >= 3.

    Returns
    -------
    ndarray, shape (...)
        ``B(s)``. Summing the right and down neighbour products counts each of
        the 2*N**2 bonds exactly once.
    """
    spins = np.asarray(spins, dtype=float)
    if spins.shape[-1] < 3 or spins.shape[-2] < 3:
        raise ValueError("Lattice must be at least 3 x 3.")
    right = np.roll(spins, shift=-1, axis=-1)
    down = np.roll(spins, shift=-1, axis=-2)
    return np.sum(spins * right + spins * down, axis=(-2, -1))


def energy(spins, J=1.0):
    """Energy ``E(s) = -J * B(s)`` (k_B = 1)."""
    return -J * bond_sum(spins)


def neighbor_sum(spins, i, j):
    """Sum of the 4 periodic nearest neighbours of site ``(i, j)``."""
    n = spins.shape[0]
    return (
        spins[(i + 1) % n, j]
        + spins[(i - 1) % n, j]
        + spins[i, (j + 1) % n]
        + spins[i, (j - 1) % n]
    )


def delta_E(spins, i, j, J=1.0):
    """Energy change from flipping spin ``(i, j)``: ``2 * J * s_ij * sum(nbrs)``."""
    return 2.0 * J * spins[i, j] * neighbor_sum(spins, i, j)


def metropolis_sweep(spins, J, T, rng):
    """One Metropolis sweep (``N**2`` attempts) performed in place.

    Each attempt chooses a site uniformly at random (with replacement), and
    accepts the flip with probability ``min(1, exp(-dE / T))``.

    Parameters
    ----------
    spins : ndarray of int, shape (N, N)
        Configuration, modified in place.
    J, T : float
        Coupling and temperature (k_B = 1).
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    int
        Number of accepted flips.
    """
    n = spins.shape[0]
    sites = rng.integers(0, n, size=(n * n, 2))
    log_u = np.log(rng.random(n * n))
    accepted = 0
    for (i, j), lu in zip(sites, log_u):
        dE = delta_E(spins, i, j, J)
        # Accept if dE <= 0, otherwise with probability exp(-dE / T).
        if dE <= 0 or lu < -dE / T:
            spins[i, j] = -spins[i, j]
            accepted += 1
    return accepted


# ---------------------------------------------------------------------------
# Exact enumeration (used by Part A tests and by Part B)
# ---------------------------------------------------------------------------


def enumerate_states(n):
    """All ``2**(n*n)`` configurations of an ``n x n`` lattice, shape (2**(n*n), n, n)."""
    n_sites = n * n
    labels = np.arange(2**n_sites)
    bits = (labels[:, None] >> np.arange(n_sites)) & 1
    return (1 - 2 * bits).astype(np.int8).reshape(-1, n, n)


class IsingModel:
    """Exactly solvable (by enumeration) Ising model on a small lattice.

    Parameters
    ----------
    n : int, optional
        Linear lattice size (keep ``n <= 4``; ``2**(n*n)`` states). Default 4.

    Attributes
    ----------
    states : ndarray of int8, shape (2**(n*n), n, n)
        All configurations.
    state_bond : ndarray, shape (2**(n*n),)
        ``B(s)`` of every configuration.
    bond_values, log_counts : ndarray
        Unique values of ``B`` and the log of how many states have each value.
        ``Z`` depends on the states only through this density of states.
    """

    def __init__(self, n=4):
        self.n = n
        self.states = enumerate_states(n)
        self.state_bond = bond_sum(self.states)
        self.bond_values, counts = np.unique(self.state_bond, return_counts=True)
        self.log_counts = np.log(counts)

    def log_partition(self, ratio):
        """Exact ``log Z`` as a function of ``ratio = J / T`` (float or ndarray)."""
        ratio = np.asarray(ratio, dtype=float)[..., None]
        return logsumexp(ratio * self.bond_values + self.log_counts, axis=-1)

    def probabilities(self, J, T):
        """Exact Boltzmann probability of every state."""
        log_w = (J / T) * self.state_bond
        p = np.exp(log_w - logsumexp(log_w))
        return p / p.sum()

    def sample(self, J, T, size, rng):
        """Draw ``size`` independent configurations from ``P(s | J, T)`` exactly."""
        idx = rng.choice(self.states.shape[0], size=size, p=self.probabilities(J, T))
        return self.states[idx]


def state_index(spins):
    """Integer label of a configuration, matching :func:`enumerate_states`."""
    bits = ((1 - spins.ravel()) // 2).astype(np.int64)
    return int(np.sum(bits << np.arange(bits.size)))


# ---------------------------------------------------------------------------
# Part A tests
# ---------------------------------------------------------------------------


def test_delta_E(n=4, n_trials=2000, seed=0):
    """Check ``delta_E == E(after) - E(before)`` exactly on random configurations."""
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        spins = rng.choice([-1, 1], size=(n, n)).astype(np.int8)
        J = rng.uniform(0.1, 3.0)
        i, j = rng.integers(0, n, size=2)
        before = energy(spins, J)
        pred = delta_E(spins, i, j, J)
        spins[i, j] *= -1
        after = energy(spins, J)
        assert np.isclose(pred, after - before, rtol=0, atol=1e-12), (pred, after - before)
    print(f"[ok] delta_E equals E(after) - E(before) on {n_trials} random flips")


def test_bond_count(n=4):
    """The all-up state must have ``B = 2 N**2`` (one count per bond)."""
    assert bond_sum(np.ones((n, n))) == 2 * n * n
    print(f"[ok] all-up {n}x{n} lattice has B = 2N^2 = {2 * n * n}")


def test_metropolis_stationary(n=3, J=1.0, T=2.5, n_sweeps=150_000, burn=2000, seed=1):
    """Compare the Metropolis visit frequencies with the exact Boltzmann law.

    A 3x3 lattice has 512 states. The comparison is made on the distribution of
    the (autocorrelated) chain, so rather than a naive chi-square test we report
    the total-variation distance and the error on the energy distribution
    (a coarser, better-populated statistic), with a tolerance set from the
    number of sweeps.
    """
    rng = np.random.default_rng(seed)
    model = IsingModel(n)
    exact = model.probabilities(J, T)

    spins = rng.choice([-1, 1], size=(n, n)).astype(np.int8)
    for _ in range(burn):
        metropolis_sweep(spins, J, T, rng)

    counts = np.zeros(exact.size)
    for _ in range(n_sweeps):
        metropolis_sweep(spins, J, T, rng)
        counts[state_index(spins)] += 1
    freq = counts / counts.sum()

    tv = 0.5 * np.abs(freq - exact).sum()

    # Energy (bond-sum) marginal: much better populated than single states.
    b_vals = model.bond_values
    exact_b = np.array([exact[model.state_bond == b].sum() for b in b_vals])
    freq_b = np.array([freq[model.state_bond == b].sum() for b in b_vals])
    max_err_b = np.abs(freq_b - exact_b).max()
    mean_b_exact = float(np.sum(exact * model.state_bond))
    mean_b_mc = float(np.sum(freq * model.state_bond))

    print(
        f"[..] Metropolis vs exact ({n}x{n}, J={J}, T={T}, {n_sweeps} sweeps): "
        f"TV = {tv:.4f}, max |dP(B)| = {max_err_b:.4f}, "
        f"<B> exact = {mean_b_exact:.3f}, MC = {mean_b_mc:.3f}"
    )
    assert max_err_b < 0.01, "energy distribution differs from Boltzmann"
    assert abs(mean_b_mc - mean_b_exact) < 0.05, "mean bond sum differs"
    assert tv < 0.05, "state distribution differs from Boltzmann"
    print("[ok] Metropolis stationary distribution matches exp(-E/T)/Z")


def test_detailed_balance(n=3, J=1.0, T=2.5, n_trials=500, seed=2):
    """Check ``P(s) * A(s->s') == P(s') * A(s'->s)`` for the acceptance rule."""
    rng = np.random.default_rng(seed)
    model = IsingModel(n)
    p = model.probabilities(J, T)
    for _ in range(n_trials):
        spins = rng.choice([-1, 1], size=(n, n)).astype(np.int8)
        i, j = rng.integers(0, n, size=2)
        dE = delta_E(spins, i, j, J)
        a_fwd = min(1.0, np.exp(-dE / T))
        s0 = state_index(spins)
        spins[i, j] *= -1
        s1 = state_index(spins)
        a_bwd = min(1.0, np.exp(dE / T))
        assert np.isclose(p[s0] * a_fwd, p[s1] * a_bwd, rtol=1e-10, atol=0)
    print(f"[ok] detailed balance holds for {n_trials} random single-flip pairs")


def run_part_a_tests():
    """Run all Part A checks."""
    test_bond_count()
    test_delta_E()
    test_detailed_balance()
    test_metropolis_stationary()


# ---------------------------------------------------------------------------
# Part B: Bayesian inference for (J, T)
# ---------------------------------------------------------------------------


class IsingPosterior:
    """Log-posterior of ``(J, T)`` given K i.i.d. observed lattices.

    ``log L(J, T) = (J/T) * sum_k B(s_k) - K * log Z(J/T)``, with uniform priors
    on a rectangle. Callable on ``theta = (J, T)``.

    Parameters
    ----------
    model : IsingModel
    data : ndarray, shape (K, n, n)
    J_bounds, T_bounds : tuple of float
        Uniform prior supports (defaults J in [0.1, 3], T in [0.5, 6]).
    """

    def __init__(self, model, data, J_bounds=(0.1, 3.0), T_bounds=(0.5, 6.0)):
        self.model = model
        self.J_bounds = J_bounds
        self.T_bounds = T_bounds
        b = bond_sum(data)
        self.K = b.size
        self.sum_bond = b.sum()

    def log_prior(self, theta):
        """0 inside the prior box, ``-inf`` outside."""
        J, T = theta
        inside = (
            self.J_bounds[0] <= J <= self.J_bounds[1]
            and self.T_bounds[0] <= T <= self.T_bounds[1]
        )
        return 0.0 if inside else -np.inf

    def log_likelihood(self, theta):
        """``(J/T) * sum B - K * log Z(J/T)``."""
        J, T = theta
        r = J / T
        return r * self.sum_bond - self.K * self.model.log_partition(r)

    def __call__(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def grid_posterior(self, n_grid=400):
        """Brute-force posterior on a (J, T) grid.

        Returns
        -------
        Js, Ts : ndarray, shape (n_grid,)
        weights : ndarray, shape (n_grid, n_grid)
            Normalised posterior mass, indexed ``[J, T]``.
        """
        Js = np.linspace(*self.J_bounds, n_grid)
        Ts = np.linspace(*self.T_bounds, n_grid)
        gJ, gT = np.meshgrid(Js, Ts, indexing="ij")
        r = gJ / gT
        log_post = r * self.sum_bond - self.K * self.model.log_partition(r)
        w = np.exp(log_post - log_post.max())
        return Js, Ts, w / w.sum()

    def grid_summary(self, n_grid=400):
        """Posterior mean/std of J, T and r = J/T from the grid."""
        Js, Ts, w = self.grid_posterior(n_grid)
        gJ, gT = np.meshgrid(Js, Ts, indexing="ij")
        out = []
        for f in (gJ, gT, gJ / gT):
            m = np.sum(w * f)
            out.append((m, np.sqrt(np.sum(w * f**2) - m**2)))
        mean, std = np.array(out).T
        return mean, std


def metropolis_hastings(
    log_prob, theta0, n_steps, n_burn, step_size, rng, scales=None, tune_interval=100
):
    """Gaussian random-walk MH; step size is tuned in burn-in only, then frozen.

    Parameters
    ----------
    log_prob : callable
    theta0 : array_like, shape (ndim,)
    n_steps, n_burn : int
        Kept and burn-in iterations.
    step_size : float
        Initial overall proposal scale.
    rng : numpy.random.Generator
    scales : array_like, shape (ndim,), optional
        Fixed per-parameter proposal multipliers (e.g. prior widths); the
        proposal std is ``step_size * scales``. Default: all ones.
    tune_interval : int
        Burn-in iterations between step-size updates.

    Returns
    -------
    chain : ndarray, shape (n_steps, ndim)
    acceptance_rate : float
        Over the post-burn-in phase only.
    step_size : float
        Final, frozen step size.
    """
    target_rate = 0.234
    theta = np.asarray(theta0, dtype=float)
    scales = np.ones(theta.size) if scales is None else np.asarray(scales, float)
    log_p = log_prob(theta)
    chain = np.empty((n_steps, theta.size))

    window_accepts = 0
    kept_accepts = 0
    for i in range(n_burn + n_steps):
        proposal = theta + step_size * scales * rng.standard_normal(theta.size)
        log_p_prop = log_prob(proposal)
        accepted = np.log(rng.uniform()) < log_p_prop - log_p
        if accepted:
            theta, log_p = proposal, log_p_prop

        if i < n_burn:  # adapt only during burn-in
            window_accepts += accepted
            if (i + 1) % tune_interval == 0:
                rate = window_accepts / tune_interval
                step_size *= np.exp(rate - target_rate)
                window_accepts = 0
        else:  # step_size is frozen from here on
            chain[i - n_burn] = theta
            kept_accepts += accepted

    return chain, kept_accepts / n_steps, step_size


def integrated_autocorr_time(chains, c=5.0):
    """Integrated autocorrelation time (Sokal windowing).

    Parameters
    ----------
    chains : ndarray, shape (n_steps, n_chains, ndim)
        Use ``chain[:, None, :]`` for a single chain.
    c : float
        Window constant.

    Returns
    -------
    ndarray, shape (ndim,)
    """
    chains = np.asarray(chains, dtype=float)
    n_steps = chains.shape[0]
    centered = chains - chains.mean(axis=0)
    n_fft = 2 ** int(np.ceil(np.log2(2 * n_steps)))
    spectrum = np.fft.rfft(centered, n=n_fft, axis=0)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), n=n_fft, axis=0)[:n_steps]
    acf = acov.mean(axis=1)
    acf /= acf[0]
    tau_m = 2.0 * np.cumsum(acf, axis=0) - 1.0
    ok = np.arange(n_steps)[:, None] >= c * tau_m
    first = np.where(ok.any(axis=0), ok.argmax(axis=0), n_steps - 1)
    return tau_m[first, np.arange(tau_m.shape[1])]


def with_ratio(samples):
    """Append the identifiable ratio ``r = J / T`` as a third column."""
    return np.column_stack([samples, samples[:, 0] / samples[:, 1]])


def summarize(samples, ess):
    """Mean, std, 95% interval and ESS of each column of ``samples``."""
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    return {
        "mean": samples.mean(axis=0),
        "std": samples.std(axis=0),
        "lo": lo,
        "hi": hi,
        "ess": np.asarray(ess),
    }


def boundary_report(samples, posterior, frac=0.02):
    """Warn if posterior mass piles up against the edge of the prior box.

    Parameters
    ----------
    samples : ndarray, shape (n, 2)
        Posterior samples of (J, T).
    posterior : IsingPosterior
    frac : float
        An edge region is the outer ``frac`` of each parameter's range.
    """
    lines = []
    for k, (name, (lo, hi)) in enumerate(
        [("J", posterior.J_bounds), ("T", posterior.T_bounds)]
    ):
        width = hi - lo
        f_lo = np.mean(samples[:, k] < lo + frac * width)
        f_hi = np.mean(samples[:, k] > hi - frac * width)
        flag = "  <-- HITS BOUNDARY, widen prior" if max(f_lo, f_hi) > 0.05 else ""
        lines.append(
            f"  {name}: {100 * f_lo:5.1f}% within {100 * frac:.0f}% of lower edge "
            f"{lo}, {100 * f_hi:5.1f}% within {100 * frac:.0f}% of upper edge {hi}{flag}"
        )
    return "\n".join(lines)


def print_table(results, truth, grid_mean, grid_std):
    """Comparison table for J, T and the identifiable ratio J/T."""
    names = ["J", "T", "J/T"]
    t_ratio = truth[0] / truth[1]
    print(f"True parameters: J={truth[0]:.3f}, T={truth[1]:.3f}, J/T={t_ratio:.3f}\n")
    header = f"{'method':<22}{'param':<6}{'mean':>8}{'std':>8}"
    header += f"{'2.5%':>8}{'97.5%':>8}{'ESS':>9}{'time[s]':>9}"
    print(header)
    print("-" * len(header))
    for j, name in enumerate(names):
        print(f"{'grid (exact)':<22}{name:<6}{grid_mean[j]:>8.4f}{grid_std[j]:>8.4f}")
    for method, res in results.items():
        for j, name in enumerate(names):
            print(
                f"{method:<22}{name:<6}{res['mean'][j]:>8.4f}{res['std'][j]:>8.4f}"
                f"{res['lo'][j]:>8.4f}{res['hi'][j]:>8.4f}"
                f"{res['ess'][j]:>9.0f}{res['time']:>9.2f}"
            )


def plot_comparison(chains, truth, grid_mean, path):
    """Scatter of joint (J, T) posterior samples per method."""
    fig, axes = plt.subplots(1, len(chains), figsize=(5 * len(chains), 4.5))
    for ax, (name, s) in zip(np.atleast_1d(axes), chains.items()):
        ax.scatter(s[::5, 0], s[::5, 1], s=3, alpha=0.3)
        ax.plot(*truth, "r*", ms=14, label="truth")
        ax.plot(grid_mean[0], grid_mean[1], "kx", ms=10, label="grid posterior mean")
        ax.set_xlabel("J")
        ax.set_ylabel("T")
        ax.set_title(name)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def run_part_b(seed=42, K=50, truth=(1.0, 2.5), use_emcee=True):
    """Simulate data exactly and compare MH, emcee and the grid posterior.

    Parameters
    ----------
    seed : int
    K : int
        Number of independent configurations drawn.
    truth : tuple of float
        True ``(J, T)``.
    use_emcee : bool
        Skip the emcee run if False (e.g. emcee not installed).
    """
    rng = np.random.default_rng(seed)
    truth = np.asarray(truth, dtype=float)

    # Exact data generation: independent of the Metropolis code of Part A.
    model = IsingModel(n=4)
    data = model.sample(truth[0], truth[1], size=K, rng=rng)
    posterior = IsingPosterior(model, data)
    grid_mean, grid_std = posterior.grid_summary()

    theta0 = np.array([1.5, 3.0])
    prior_widths = np.array(
        [posterior.J_bounds[1] - posterior.J_bounds[0],
         posterior.T_bounds[1] - posterior.T_bounds[0]]
    )
    results, chains = {}, {}

    # (1) hand-written MH
    t0 = time.perf_counter()
    chain, acc, step = metropolis_hastings(
        posterior, theta0, n_steps=40000, n_burn=3000, step_size=0.05,
        rng=rng, scales=prior_widths,
    )
    elapsed = time.perf_counter() - t0
    aug = with_ratio(chain)
    tau = integrated_autocorr_time(aug[:, None, :])
    results["hand-written MH"] = summarize(aug, aug.shape[0] / tau)
    results["hand-written MH"]["time"] = elapsed
    chains["hand-written MH"] = chain
    print(f"MH acceptance rate: {acc:.2f} (frozen step size {step:.4f})")

    # (2) emcee
    if use_emcee:
        import emcee

        n_walkers, n_steps, n_burn = 16, 2500, 500
        start = theta0 + 1e-2 * rng.standard_normal((n_walkers, 2))
        sampler = emcee.EnsembleSampler(n_walkers, 2, posterior)
        t0 = time.perf_counter()
        sampler.run_mcmc(start, n_burn + n_steps, progress=False)
        elapsed = time.perf_counter() - t0
        wc = sampler.get_chain(discard=n_burn)  # (steps, walkers, 2)
        r = (wc[..., 0] / wc[..., 1])[..., None]
        wc_aug = np.concatenate([wc, r], axis=-1)
        tau = integrated_autocorr_time(wc_aug)
        flat = wc.reshape(-1, 2)
        flat_aug = wc_aug.reshape(-1, 3)
        results["emcee"] = summarize(flat_aug, flat_aug.shape[0] / tau)
        results["emcee"]["time"] = elapsed
        chains["emcee"] = flat

    # Compare
    print_table(results, truth, grid_mean, grid_std)
    print("\nPrior-boundary check (hand-written MH samples):")
    print(boundary_report(chains["hand-written MH"], posterior))
    plot_comparison(chains, truth, grid_mean, "ising_posterior.png")
    print("\nSaved figure to ising_posterior.png")


def main():
    """Run the Part A checks, then the Part B comparison."""
    print("=== Part A: spin-level checks ===")
    run_part_a_tests()
    print("\n=== Part B: Bayesian inference ===")
    try:
        import emcee  # noqa: F401
        have_emcee = True
    except ImportError:
        print("emcee not installed; running MH and grid only.")
        have_emcee = False
    run_part_b(use_emcee=have_emcee)


if __name__ == "__main__":
    main()