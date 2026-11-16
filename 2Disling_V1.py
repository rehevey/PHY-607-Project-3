"""Bayesian parameter estimation for a small 2D Ising model.

The model is a square lattice of ``n x n`` spins ``s_i`` in ``{-1, +1}`` with
periodic boundaries and probability mass function

.. math::

    p(s \\mid \\beta, h) = \\frac{1}{Z(\\beta, h)}
        \\exp\\Big(\\beta \\sum_{\\langle i, j \\rangle} s_i s_j
        + h \\sum_i s_i\\Big),

where ``beta`` is the (dimensionless) coupling / inverse temperature and ``h``
is the external field.

For general lattices the partition function ``Z`` is intractable, which makes
the likelihood "doubly intractable". For a small lattice (default ``4 x 4``,
i.e. 65536 states) ``Z`` can be computed *exactly* by enumeration, so we can
compare two MCMC samplers on the *same*, exact posterior:

1. a hand written random-walk Metropolis-Hastings sampler, and
2. the affine-invariant ensemble sampler from the ``emcee`` library.

A brute-force grid evaluation of the posterior serves as ground truth.

Run as a script to execute the full comparison::

    python ising_inference.py
"""

import time

import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp


def enumerate_states(n):
    """Enumerate every spin configuration of an ``n x n`` lattice.

    Parameters
    ----------
    n : int
        Linear size of the lattice.

    Returns
    -------
    ndarray of int8, shape (2**(n*n), n, n)
        All configurations with entries in ``{-1, +1}``.
    """
    n_sites = n * n
    # Each integer in [0, 2**n_sites) encodes one configuration in binary
    labels = np.arange(2**n_sites)
    bits = (labels[:, None] >> np.arange(n_sites)) & 1
    # Map bit {0, 1} -> spin {+1, -1}
    spins = (1 - 2 * bits).astype(np.int8)
    return spins.reshape(-1, n, n)


def sufficient_statistics(spins):
    """Compute the Ising sufficient statistics of spin configuration(s).

    Parameters
    ----------
    spins : ndarray, shape (..., n, n)
        One or more lattices with entries in ``{-1, +1}``. The last two axes
        are the lattice axes; periodic boundaries are assumed.

    Returns
    -------
    bond_sum : ndarray, shape (...)
        Sum of ``s_i * s_j`` over all nearest-neighbour bonds.
    magnetization : ndarray, shape (...)
        Sum of all spins.
    """
    spins = np.asarray(spins, dtype=float)
    # np.roll gives the periodic right / down neighbour of every site, so each
    # bond is counted exactly once by multiplying elementwise
    right = np.roll(spins, shift=-1, axis=-1)
    down = np.roll(spins, shift=-1, axis=-2)
    bond_sum = np.sum(spins * right + spins * down, axis=(-2, -1))
    magnetization = np.sum(spins, axis=(-2, -1))
    return bond_sum, magnetization


class IsingModel:
    """Exactly solvable (by enumeration) 2D Ising model on a small lattice.

    Parameters
    ----------
    n : int, optional
        Linear lattice size. Memory / time scale as ``2**(n*n)``, so keep
        ``n <= 5``. Default is 4.

    Attributes
    ----------
    n : int
        Linear lattice size.
    states : ndarray of int8, shape (2**(n*n), n, n)
        All spin configurations.
    bond_values, mag_values : ndarray, shape (n_unique,)
        Unique values of the sufficient statistics (the "density of states"
        support). Using these instead of all states greatly speeds up ``Z``.
    log_counts : ndarray, shape (n_unique,)
        Log of the number of configurations sharing each statistic pair.
    """

    def __init__(self, n=4):
        self.n = n
        self.states = enumerate_states(n)

        # Collapse the 2**(n*n) states onto unique (bond_sum, magnetization)
        # pairs; Z only depends on the pairs and their multiplicities
        bond, mag = sufficient_statistics(self.states)
        pairs, counts = np.unique(
            np.column_stack([bond, mag]), axis=0, return_counts=True
        )
        self.bond_values = pairs[:, 0]
        self.mag_values = pairs[:, 1]
        self.log_counts = np.log(counts)

        # Per-state statistics are kept for exact sampling
        self._state_bond = bond
        self._state_mag = mag

    def log_partition(self, beta, h):
        """Exact log partition function ``log Z(beta, h)``.

        Parameters
        ----------
        beta, h : float or ndarray
            Parameters. Arrays must be broadcastable against each other.

        Returns
        -------
        float or ndarray
            ``log Z`` with the broadcast shape of the inputs.
        """
        beta = np.asarray(beta, dtype=float)[..., None]
        h = np.asarray(h, dtype=float)[..., None]
        # Log-sum-exp over the (weighted) unique statistic pairs
        exponent = beta * self.bond_values + h * self.mag_values
        return logsumexp(exponent + self.log_counts, axis=-1)

    def sample(self, beta, h, size, rng):
        """Draw exact i.i.d. configurations from the model.

        Parameters
        ----------
        beta, h : float
            Model parameters.
        size : int
            Number of lattices to draw.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        ndarray of int8, shape (size, n, n)
            Independent spin configurations.
        """
        # Boltzmann probability of every one of the 2**(n*n) states
        log_weights = beta * self._state_bond + h * self._state_mag
        probs = np.exp(log_weights - logsumexp(log_weights))
        probs /= probs.sum()  # guard against round-off
        idx = rng.choice(probs.size, size=size, p=probs)
        return self.states[idx]


class IsingPosterior:
    """Log-posterior of ``(beta, h)`` given observed Ising lattices.

    A uniform prior on a rectangular box is used. Because the Ising model is
    an exponential family, the likelihood only depends on the summed
    sufficient statistics of the data.

    Parameters
    ----------
    model : IsingModel
        Model used to evaluate the exact partition function.
    data : ndarray, shape (n_obs, n, n)
        Observed i.i.d. spin configurations.
    beta_bounds : tuple of float, optional
        Support of the uniform prior on ``beta``.
    h_bounds : tuple of float, optional
        Support of the uniform prior on ``h``.

    Attributes
    ----------
    n_obs : int
        Number of observed lattices.
    sum_bond, sum_mag : float
        Sufficient statistics summed over all observed lattices.
    """

    def __init__(self, model, data, beta_bounds=(0.0, 1.0), h_bounds=(-1.0, 1.0)):
        self.model = model
        self.beta_bounds = beta_bounds
        self.h_bounds = h_bounds

        bond, mag = sufficient_statistics(data)
        self.n_obs = bond.size
        self.sum_bond = bond.sum()
        self.sum_mag = mag.sum()

    def log_prior(self, theta):
        """Log of the (unnormalised) uniform box prior.

        Parameters
        ----------
        theta : array_like, shape (2,)
            Parameter vector ``(beta, h)``.

        Returns
        -------
        float
            ``0`` inside the prior box and ``-inf`` outside.
        """
        beta, h = theta
        inside = (
            self.beta_bounds[0] <= beta <= self.beta_bounds[1]
            and self.h_bounds[0] <= h <= self.h_bounds[1]
        )
        return 0.0 if inside else -np.inf

    def log_likelihood(self, theta):
        """Exact log-likelihood of the data.

        Parameters
        ----------
        theta : array_like, shape (2,)
            Parameter vector ``(beta, h)``.

        Returns
        -------
        float
            ``beta * sum_bond + h * sum_mag - n_obs * log Z(beta, h)``.
        """
        beta, h = theta
        return (
            beta * self.sum_bond
            + h * self.sum_mag
            - self.n_obs * self.model.log_partition(beta, h)
        )

    def __call__(self, theta):
        """Unnormalised log-posterior (prior + likelihood).

        Parameters
        ----------
        theta : array_like, shape (2,)
            Parameter vector ``(beta, h)``.

        Returns
        -------
        float
            Log-posterior, or ``-inf`` outside the prior support.
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def grid_summary(self, n_grid=300):
        """Posterior mean / std from a brute-force grid (ground truth).

        Parameters
        ----------
        n_grid : int, optional
            Number of grid points per parameter.

        Returns
        -------
        mean, std : ndarray, shape (2,)
            Posterior mean and standard deviation of ``(beta, h)``.
        """
        betas = np.linspace(*self.beta_bounds, n_grid)
        hs = np.linspace(*self.h_bounds, n_grid)
        grid_b, grid_h = np.meshgrid(betas, hs, indexing="ij")

        # Fully vectorised log-posterior on the grid (prior is flat)
        log_post = (
            grid_b * self.sum_bond
            + grid_h * self.sum_mag
            - self.n_obs * self.model.log_partition(grid_b, grid_h)
        )
        weights = np.exp(log_post - log_post.max())
        weights /= weights.sum()

        # Moments from the discrete posterior weights
        mean = np.array([np.sum(weights * grid_b), np.sum(weights * grid_h)])
        second = np.array([np.sum(weights * grid_b**2), np.sum(weights * grid_h**2)])
        return mean, np.sqrt(second - mean**2)


def metropolis_hastings(
    log_prob, theta0, n_steps, n_burn, step_size, rng, tune_interval=100
):
    """Random-walk Metropolis-Hastings with burn-in step-size tuning.

    Parameters
    ----------
    log_prob : callable
        Function mapping a parameter vector to the unnormalised log density.
    theta0 : array_like, shape (ndim,)
        Initial state of the chain.
    n_steps : int
        Number of post burn-in samples to keep.
    n_burn : int
        Number of burn-in iterations (discarded). The proposal scale is
        adapted during burn-in only, so the kept chain is a valid Markov
        chain with a fixed kernel.
    step_size : float
        Initial standard deviation of the isotropic Gaussian proposal.
    rng : numpy.random.Generator
        Random number generator.
    tune_interval : int, optional
        Number of burn-in iterations between step-size updates.

    Returns
    -------
    chain : ndarray, shape (n_steps, ndim)
        Post burn-in samples.
    acceptance_rate : float
        Fraction of accepted proposals after burn-in.
    step_size : float
        Final (tuned) proposal scale.
    """
    target_rate = 0.234  # asymptotically optimal rate for random-walk MH
    theta = np.asarray(theta0, dtype=float)
    log_p = log_prob(theta)
    chain = np.empty((n_steps, theta.size))

    window_accepts = 0
    kept_accepts = 0
    for i in range(n_burn + n_steps):
        # Symmetric Gaussian proposal -> Hastings ratio reduces to p'/p
        proposal = theta + step_size * rng.standard_normal(theta.size)
        log_p_prop = log_prob(proposal)

        # Accept with probability min(1, p'/p); done in log space for
        # numerical stability (log(u) < -inf is False, so -inf is rejected)
        accepted = np.log(rng.uniform()) < log_p_prop - log_p
        if accepted:
            theta, log_p = proposal, log_p_prop

        if i < n_burn:
            # Adapt the proposal scale towards the target acceptance rate
            window_accepts += accepted
            if (i + 1) % tune_interval == 0:
                rate = window_accepts / tune_interval
                step_size *= np.exp(rate - target_rate)
                window_accepts = 0
        else:
            chain[i - n_burn] = theta
            kept_accepts += accepted

    return chain, kept_accepts / n_steps, step_size


def summarize(samples, ess):
    """Summarise posterior samples.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, ndim)
        Flattened posterior samples.
    ess : ndarray, shape (ndim,)
        Effective sample size for each parameter.

    Returns
    -------
    dict
        Posterior ``mean``, ``std``, 95% interval bounds ``lo`` / ``hi`` and
        ``ess``, each an array of shape ``(ndim,)``.
    """
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    return {
        "mean": samples.mean(axis=0),
        "std": samples.std(axis=0),
        "lo": lo,
        "hi": hi,
        "ess": np.asarray(ess),
    }


def print_table(results, truth, grid_mean, grid_std):
    """Print a comparison table of the estimation methods.

    Parameters
    ----------
    results : dict of str to dict
        Mapping from method name to the output of :func:`summarize`, with an
        additional ``"time"`` entry (seconds).
    truth : array_like, shape (2,)
        True parameters used to simulate the data.
    grid_mean, grid_std : ndarray, shape (2,)
        Ground-truth posterior moments from the grid.
    """
    names = ["beta", "h"]
    print(f"True parameters: beta={truth[0]:.3f}, h={truth[1]:.3f}\n")
    header = f"{'method':<22}{'param':<6}{'mean':>8}{'std':>8}"
    header += f"{'2.5%':>8}{'97.5%':>8}{'ESS':>9}{'time[s]':>9}"
    print(header)
    print("-" * len(header))
    for j, name in enumerate(names):
        print(
            f"{'grid (exact)':<22}{name:<6}{grid_mean[j]:>8.4f}{grid_std[j]:>8.4f}"
        )
    for method, res in results.items():
        for j, name in enumerate(names):
            print(
                f"{method:<22}{name:<6}{res['mean'][j]:>8.4f}{res['std'][j]:>8.4f}"
                f"{res['lo'][j]:>8.4f}{res['hi'][j]:>8.4f}"
                f"{res['ess'][j]:>9.0f}{res['time']:>9.2f}"
            )


def plot_comparison(chains, truth, grid_mean, path):
    """Plot the joint posterior samples from each method.

    Parameters
    ----------
    chains : dict of str to ndarray
        Mapping from method name to samples of shape ``(n_samples, 2)``.
    truth : array_like, shape (2,)
        True parameters.
    grid_mean : ndarray, shape (2,)
        Grid-based posterior mean.
    path : str
        Output image path.
    """
    fig, axes = plt.subplots(1, len(chains), figsize=(5 * len(chains), 4.5))
    for ax, (name, samples) in zip(np.atleast_1d(axes), chains.items()):
        ax.scatter(samples[::5, 0], samples[::5, 1], s=3, alpha=0.3)
        ax.plot(*truth, "r*", ms=14, label="truth")
        ax.plot(*grid_mean, "kx", ms=10, label="grid posterior mean")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$h$")
        ax.set_title(name)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def integrated_autocorr_time(chains, c=5.0):
    """Integrated autocorrelation time of MCMC chains (Sokal windowing).

    Parameters
    ----------
    chains : ndarray, shape (n_steps, n_chains, ndim)
        Post burn-in samples. A single chain should be passed with
        ``n_chains = 1``, e.g. ``chain[:, None, :]``.
    c : float, optional
        Window constant: the autocorrelation sum is truncated at the first
        lag ``M`` with ``M >= c * tau(M)``.

    Returns
    -------
    ndarray, shape (ndim,)
        Integrated autocorrelation time for each parameter.
    """
    chains = np.asarray(chains, dtype=float)
    n_steps = chains.shape[0]
    centered = chains - chains.mean(axis=0)

    # Autocovariance via FFT; zero-padding to >= 2n avoids circular wrap
    n_fft = 2 ** int(np.ceil(np.log2(2 * n_steps)))
    spectrum = np.fft.rfft(centered, n=n_fft, axis=0)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), n=n_fft, axis=0)[:n_steps]

    # Average over chains, then normalise so the lag-0 value is 1
    acf = acov.mean(axis=1)
    acf /= acf[0]

    # tau(M) = 1 + 2 * sum_{k=1}^{M} rho_k, evaluated for every window M
    tau_m = 2.0 * np.cumsum(acf, axis=0) - 1.0

    # Pick, per parameter, the first window satisfying the Sokal criterion
    ok = np.arange(n_steps)[:, None] >= c * tau_m
    first = np.where(ok.any(axis=0), ok.argmax(axis=0), n_steps - 1)
    return tau_m[first, np.arange(tau_m.shape[1])]


def main(seed=42):
    """Simulate Ising data and compare two posterior samplers.

    Parameters
    ----------
    seed : int, optional
        Seed of the random number generator.
    """
    rng = np.random.default_rng(seed)
    truth = np.array([0.35, 0.10])

    # --- simulate data from the exact model -------------------------------
    model = IsingModel(n=4)
    data = model.sample(*truth, size=50, rng=rng)
    posterior = IsingPosterior(model, data)
    grid_mean, grid_std = posterior.grid_summary()

    theta0 = np.array([0.5, 0.0])
    results, chains = {}, {}

    # --- (1) hand-written Metropolis-Hastings ------------------------------
    t0 = time.perf_counter()
    chain, acc, step = metropolis_hastings(
        posterior, theta0, n_steps=40000, n_burn=2000, step_size=0.1, rng=rng
    )
    elapsed = time.perf_counter() - t0
    tau = integrated_autocorr_time(chain[:, None, :])
    results["hand-written MH"] = summarize(chain, chain.shape[0] / tau)
    results["hand-written MH"]["time"] = elapsed
    chains["hand-written MH"] = chain
    print(f"MH acceptance rate: {acc:.2f} (tuned step size {step:.3f})")

    # --- (2) emcee ensemble sampler -----------------------------------------
    n_walkers, n_steps, n_burn = 16, 2500, 500
    # Start walkers in a small ball around the initial guess
    start = theta0 + 1e-2 * rng.standard_normal((n_walkers, 2))
    start[:, 0] = np.abs(start[:, 0])  # keep inside the prior support
    sampler = emcee.EnsembleSampler(n_walkers, 2, posterior)
    t0 = time.perf_counter()
    sampler.run_mcmc(start, n_burn + n_steps, progress=False)
    elapsed = time.perf_counter() - t0
    # Chain shape (steps, walkers, ndim); same ESS estimator as for MH
    walker_chain = sampler.get_chain(discard=n_burn)
    tau = integrated_autocorr_time(walker_chain)
    flat = walker_chain.reshape(-1, walker_chain.shape[-1])
    results["emcee"] = summarize(flat, flat.shape[0] / tau)
    results["emcee"]["time"] = elapsed
    chains["emcee"] = flat

    # --- compare ---------------------------------------------------------------
    print_table(results, truth, grid_mean, grid_std)
    plot_comparison(chains, truth, grid_mean, "ising_posterior.png")
    print("\nSaved figure to ising_posterior.png")


if __name__ == "__main__":
    main()