"""Test suite written from the plan (tests A-F), not from the implementation.

Conventions
-----------
* M is the mean spin per site (so |M| <= 1); E/N^2 is energy per site.
* The statistics (energy, bond sums) used to judge the sampler are computed
  here independently of the module under test.
* Slow tests are parametrized over SEEDS (see conftest.py). A 3-sigma test
  fails by chance about 1 time in 370 per quantity, so one marginal miss
  means rerun with other seeds, not panic.
"""

import math

import numpy as np
import pytest
from scipy import stats
from scipy.special import logsumexp

from conftest import SEEDS, load_ising

ising = load_ising()
slow = pytest.mark.slow

J_BOUNDS = (0.1, 3.0)
T_BOUNDS = (0.5, 6.0)
TC = 2.0 / math.log(1.0 + math.sqrt(2.0))

MODEL3 = ising.IsingModel(3)
MODEL4 = ising.IsingModel(4)


# ---------------------------------------------------------------------------
# Independent helpers
# ---------------------------------------------------------------------------


def bonds(s):
    """Sum of s_i s_j over the 2N^2 periodic bonds (batch-aware, independent)."""
    s = np.asarray(s, dtype=float)
    return np.sum(s * np.roll(s, -1, axis=-1) + s * np.roll(s, -1, axis=-2), axis=(-2, -1))


def checkerboard(n):
    i, j = np.indices((n, n))
    return ((-1) ** (i + j)).astype(np.int8)


def starts(n, rng):
    """Over-dispersed starts: all-up, all-down, random, checkerboard."""
    out = {
        "up": np.ones((n, n), np.int8),
        "down": -np.ones((n, n), np.int8),
        "random": rng.choice([-1, 1], size=(n, n)).astype(np.int8),
    }
    if n % 2 == 0:
        out["checker"] = checkerboard(n)
    return out


def run_chain(init, J, T, n_sweeps, rng, n_burn=0):
    """Metropolis chain; returns (E/N^2, M) per sweep after n_burn sweeps."""
    s = init.astype(np.int8).copy()
    n = s.shape[0]
    for _ in range(n_burn):
        ising.metropolis_sweep(s, J, T, rng)
    E = np.empty(n_sweeps)
    M = np.empty(n_sweeps)
    for k in range(n_sweeps):
        ising.metropolis_sweep(s, J, T, rng)
        E[k] = -J * bonds(s) / n**2
        M[k] = s.mean()
    return E, M


def tau_of(arr):
    """Integrated autocorrelation time of arr, shape (steps, chains)."""
    arr = np.asarray(arr, float)
    return max(1.0, float(ising.integrated_autocorr_time(arr[:, :, None])[0]))


def tau1(x):
    return tau_of(np.asarray(x)[:, None])


def se_mean(x):
    x = np.asarray(x)
    return math.sqrt(x.var() * tau1(x) / x.size)


def pooled_mean_se(chains):
    means = [np.mean(x) for x in chains]
    ses = [se_mean(x) for x in chains]
    return float(np.mean(means)), float(np.sqrt(np.sum(np.square(ses))) / len(ses))


def split_rhat(x):
    """Split Gelman-Rubin R-hat for x of shape (n_chains, n_steps)."""
    x = np.asarray(x, float)
    h = x.shape[1] // 2
    c = np.concatenate([x[:, :h], x[:, h : 2 * h]], axis=0)
    W = c.var(axis=1, ddof=1).mean()
    B = h * c.mean(axis=1).var(ddof=1)
    var_plus = (h - 1) / h * W + B / h
    return math.sqrt(var_plus / W)


def exact_moments(model, n, J, T):
    p = model.probabilities(J, T)
    st = model.states
    E = -J * bonds(st) / n**2
    M = st.reshape(len(st), -1).mean(axis=1)
    return {"E": float(p @ E), "absM": float(p @ np.abs(M)), "M2": float(p @ M**2)}


# ---------------------------------------------------------------------------
# A. Deterministic correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4, 5, 8])
@pytest.mark.parametrize("J", [0.5, 1.0, 2.0])
def test_A1_all_up_energy(n, J):
    assert ising.energy(np.ones((n, n), np.int8), J) == -2 * J * n**2


@pytest.mark.parametrize("n", [4, 6, 8])
@pytest.mark.parametrize("J", [0.5, 1.0, 2.0])
def test_A1_checkerboard_energy(n, J):
    assert ising.energy(checkerboard(n), J) == 2 * J * n**2


def test_A2_delta_E_matches_energy_difference():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        n = int(rng.choice([3, 4, 5, 6]))
        J = float(rng.choice([0.5, 1.0, 2.0]))  # exactly representable
        s = rng.choice([-1, 1], size=(n, n)).astype(np.int8)
        i, j = (int(v) for v in rng.integers(0, n, size=2))
        before = ising.energy(s, J)
        pred = ising.delta_E(s, i, j, J)
        s[i, j] *= -1
        after = ising.energy(s, J)
        assert pred == after - before


def test_A3_global_flip_and_translation_invariance():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.choice([3, 4, 5, 6]))
        s = rng.choice([-1, 1], size=(n, n)).astype(np.int8)
        e0 = ising.energy(s, 1.0)
        assert ising.energy(-s, 1.0) == e0
        for _ in range(3):
            a, b = (int(v) for v in rng.integers(0, n, size=2))
            assert ising.energy(np.roll(s, (a, b), axis=(0, 1)), 1.0) == e0


def test_A_state_index_roundtrip():
    rng = np.random.default_rng(2)
    states = ising.enumerate_states(3)
    for _ in range(200):
        s = rng.choice([-1, 1], size=(3, 3)).astype(np.int8)
        assert np.array_equal(states[ising.state_index(s)], s)


def test_A_bond_sum_requires_n_ge_3():
    with pytest.raises(ValueError):
        ising.bond_sum(np.ones((2, 2)))


# ---------------------------------------------------------------------------
# B. Known analytic answers
# ---------------------------------------------------------------------------


@slow
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("T", [2.0, 4.0])
def test_B1_single_attempt_acceptance(seed, T):
    """1x1 lattice is its own 4 neighbours, so dE = 8J. J = 0.5 gives dE = 4."""
    rng = np.random.default_rng(100 + seed)
    n_trials = 100_000
    s = np.ones((1, 1), np.int8)
    assert ising.delta_E(s, 0, 0, 0.5) == 4.0
    flips = 0
    for _ in range(n_trials):
        s[0, 0] = 1
        flips += ising.metropolis_sweep(s, 0.5, T, rng)
    p = math.exp(-4.0 / T)
    sigma = math.sqrt(p * (1 - p) / n_trials)
    assert abs(flips / n_trials - p) < 3 * sigma


def test_B1_downhill_always_accepted():
    """dE = -4 (J = -0.5 on the 1x1 lattice) must be accepted every time."""
    rng = np.random.default_rng(5)
    s = np.ones((1, 1), np.int8)
    for _ in range(2000):
        s[0, 0] = 1
        assert ising.metropolis_sweep(s, -0.5, 2.0, rng) == 1


@slow
@pytest.mark.parametrize("seed", SEEDS)
def test_B2_full_distribution_3x3(seed):
    J, T = 1.0, 2.0
    rng = np.random.default_rng(200 + seed)
    p = MODEL3.probabilities(J, T)
    s = rng.choice([-1, 1], size=(3, 3)).astype(np.int8)
    for _ in range(2000):
        ising.metropolis_sweep(s, J, T, rng)
    n_sweeps = 250_000
    idx = np.empty(n_sweeps, dtype=np.int64)
    b = np.empty(n_sweeps)
    m = np.empty(n_sweeps)
    for k in range(n_sweeps):
        ising.metropolis_sweep(s, J, T, rng)
        idx[k] = ising.state_index(s)
        b[k] = bonds(s)
        m[k] = s.mean()
    thin = int(math.ceil(4 * max(tau1(b), tau1(m))))
    idx = idx[::thin]
    n = idx.size
    counts = np.bincount(idx, minlength=p.size)
    # Merge states into bins with expected count >= 5.
    bins, cur, acc = [], [], 0.0
    for k in np.argsort(p):
        cur.append(k)
        acc += p[k] * n
        if acc >= 5:
            bins.append(cur)
            cur, acc = [], 0.0
    if cur:
        bins[-1] += cur
    obs = np.array([counts[b_].sum() for b_ in bins], float)
    exp = np.array([p[b_].sum() * n for b_ in bins])
    chi2 = ((obs - exp) ** 2 / exp).sum()
    pval = stats.chi2.sf(chi2, len(bins) - 1)
    assert pval > 0.01, (pval, thin, n, len(bins))


@slow
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("T", [1.5, 2.27, 4.0])
def test_B3_exact_moments_4x4(seed, T):
    J, n = 1.0, 4
    rng = np.random.default_rng(300 + seed)
    exact = exact_moments(MODEL4, n, J, T)
    Es, Ms = [], []
    for init in starts(n, rng).values():
        E, M = run_chain(init, J, T, 30_000, rng, n_burn=1000)
        Es.append(E)
        Ms.append(M)
    obs = {
        "E": [E for E in Es],
        "absM": [np.abs(M) for M in Ms],
        "M2": [M**2 for M in Ms],
    }
    for key, chains in obs.items():
        mean, se = pooled_mean_se(chains)
        assert abs(mean - exact[key]) < 3 * se, (key, T, mean, exact[key], se)


@pytest.mark.parametrize("seed", SEEDS)
def test_B4_low_temperature_limit(seed):
    rng = np.random.default_rng(400 + seed)
    n = 16
    E, M = run_chain(np.ones((n, n), np.int8), 1.0, 0.3, 1000, rng, n_burn=200)
    assert abs(E.mean() - (-2.0)) < 0.05
    assert abs(np.abs(M).mean() - 1.0) < 0.05


@pytest.mark.parametrize("seed", SEEDS)
def test_B4_high_temperature_limit(seed):
    rng = np.random.default_rng(450 + seed)
    n = 16
    init = rng.choice([-1, 1], size=(n, n)).astype(np.int8)
    E, M = run_chain(init, 1.0, 50.0, 2000, rng, n_burn=200)
    assert abs(E.mean() - 0.0) < 0.05
    assert abs(np.abs(M).mean() - 0.0) < 0.1


@slow
@pytest.mark.parametrize("seed", SEEDS[:1])
def test_B5_onsager_crossing_and_steepening(seed):
    rng = np.random.default_rng(500 + seed)
    sizes = [8, 16, 32]
    Ts = np.round(np.arange(2.0, 2.65, 0.1), 2)
    table = {}
    for n in sizes:
        row = []
        for T in Ts:
            s = np.ones((n, n), np.int8)
            for _ in range(400):
                ising.metropolis_sweep(s, 1.0, float(T), rng)
            vals = []
            for _ in range(800):
                ising.metropolis_sweep(s, 1.0, float(T), rng)
                vals.append(abs(s.mean()))
            row.append(np.mean(vals))
        table[n] = np.array(row)
    drop = [table[n][0] - table[n][-1] for n in sizes]
    assert drop[0] < drop[1] < drop[2], drop  # |M|(T) steepens with N
    for small, large in [(8, 16), (16, 32)]:
        d = table[small] - table[large]  # negative below T_c, positive above
        sign_change = np.where((d[:-1] < 0) & (d[1:] >= 0))[0]
        assert sign_change.size > 0, (small, large, d)
        k = sign_change[0]
        Tx = Ts[k] + (0 - d[k]) / (d[k + 1] - d[k]) * (Ts[k + 1] - Ts[k])
        assert abs(Tx - TC) < 0.15, (small, large, Tx)


# ---------------------------------------------------------------------------
# C. Symmetry and degeneracy
# ---------------------------------------------------------------------------


@slow
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("T", [1.5, 3.0])
def test_C1_M_symmetry_pooled_up_down(seed, T):
    rng = np.random.default_rng(600 + seed)
    n = 4
    chains = []
    for key in ("up", "down"):
        _, M = run_chain(starts(n, rng)[key], 1.0, T, 40_000, rng, n_burn=1000)
        chains.append(M)
    crossed = all((np.sign(M).min() < 0 < np.sign(M).max()) for M in chains)
    if not crossed:
        pytest.skip("chains did not cross sectors; expected below T_c (P3), see D2")
    pooled = np.concatenate(chains)
    n_eff = sum(x.size / tau1(x) for x in chains)
    mean_se = pooled.std() / math.sqrt(n_eff)
    assert abs(pooled.mean()) < 3 * mean_se
    assert abs(stats.skew(pooled)) < 3 * math.sqrt(6.0 / n_eff)


def _prior_samples(rng, n=1_000_000):
    J = rng.uniform(*J_BOUNDS, n)
    T = rng.uniform(*T_BOUNDS, n)
    return J, T


def make_data(seed, K, J=1.0, T=2.5):
    rng = np.random.default_rng(seed)
    return MODEL4.sample(J, T, size=K, rng=rng)


def run_mh(post, seed, n_steps=150_000, n_burn=5000):
    rng = np.random.default_rng(seed)
    widths = np.array([J_BOUNDS[1] - J_BOUNDS[0], T_BOUNDS[1] - T_BOUNDS[0]])
    chain, acc, _ = ising.metropolis_hastings(
        post, np.array([1.5, 3.0]), n_steps=n_steps, n_burn=n_burn,
        step_size=0.05, rng=rng, scales=widths,
    )
    return chain, acc


@slow
@pytest.mark.parametrize("seed", SEEDS)
def test_C2_ridge(seed):
    data = make_data(700 + seed, K=20)
    post = ising.IsingPosterior(MODEL4, data)
    chain, _ = run_mh(post, 710 + seed)
    J, T = chain[:, 0], chain[:, 1]
    r = J / T
    pJ, pT = _prior_samples(np.random.default_rng(0))
    assert r.std() < 0.25 * (pJ / pT).std()
    assert np.corrcoef(J, T)[0, 1] > 0.9
    assert J.std() > 0.5 * pJ.std()


@pytest.mark.parametrize("J,T", [(1.0, 2.5), (0.3, 4.0), (2.2, 0.9), (1.7, 1.7)])
def test_C3_likelihood_scale_invariance(J, T):
    data = make_data(800, K=10)
    post = ising.IsingPosterior(MODEL4, data)
    assert np.isclose(post.log_likelihood((J, T)), post.log_likelihood((2 * J, 2 * T)),
                      rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# D. Convergence diagnostics
# ---------------------------------------------------------------------------

BURN = 500
N_RUN = 5000
_cache = {}


def overdispersed(n, T, n_run, seed):
    key = (n, T, n_run, seed)
    if key not in _cache:
        rng = np.random.default_rng(900 + seed)
        out = {}
        for name, init in starts(n, rng).items():
            out[name] = run_chain(init, 1.0, T, n_run, rng, n_burn=0)
        _cache[key] = out
    return _cache[key]


def _stack(out, which, burn=BURN):
    idx = {"E": 0, "M": 1}[which if which != "absM" else "M"]
    x = np.array([v[idx][burn:] for v in out.values()])
    return np.abs(x) if which == "absM" else x


@slow
@pytest.mark.parametrize("seed", SEEDS[:1])
def test_D_burn_in_justification(seed):
    """Burn-in from the all-up / all-down relaxation time, not a guess."""
    out = overdispersed(8, 3.5, N_RUN + BURN, seed)
    late = np.concatenate([out["up"][0][N_RUN // 2:], out["down"][0][N_RUN // 2:]]).mean()
    w = 25
    ok_times = []
    for key in ("up", "down"):
        roll = np.convolve(out[key][0], np.ones(w) / w, mode="valid")
        ok_times.append(int(np.argmax(np.abs(roll - late) < 0.05)))
    t_relax = max(ok_times)
    print(f"burn-in justification: up/down E traces within 0.05 of late mean by sweep "
          f"{t_relax}; BURN = {BURN}")
    assert 4 * t_relax <= BURN


@slow
@pytest.mark.parametrize("seed", SEEDS[:1])
def test_D1_trace_plots(seed, tmp_path):
    import matplotlib.pyplot as plt

    out = overdispersed(8, 3.5, N_RUN + BURN, seed)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    for name, (E, M) in out.items():
        ax[0].plot(E[:2000], lw=0.6, label=name)
        ax[1].plot(np.abs(M[:2000]), lw=0.6, label=name)
    ax[0].set_ylabel("E/N^2")
    ax[1].set_ylabel("|M|")
    ax[0].legend()
    path = tmp_path / "traces.png"
    fig.savefig(path)
    assert path.exists()


@slow
@pytest.mark.parametrize("seed", SEEDS)
def test_D2_rhat_above_Tc(seed):
    out = overdispersed(8, 3.5, N_RUN + BURN, seed)
    for which in ("E", "absM", "M"):
        assert split_rhat(_stack(out, which)) < 1.05, which


@slow
@pytest.mark.parametrize("seed", SEEDS[:1])
def test_D2_rhat_planned_failure_below_Tc(seed):
    """P3: up/down chains on a 16x16 grid at T = 1.5 never cross."""
    rng = np.random.default_rng(950 + seed)
    n, T = 16, 1.5
    st = starts(n, rng)
    E_u, M_u = run_chain(st["up"], 1.0, T, 3000, rng, n_burn=200)
    E_d, M_d = run_chain(st["down"], 1.0, T, 3000, rng, n_burn=200)
    assert M_u.min() > 0 and M_d.max() < 0  # P3: opposite sectors persist
    assert split_rhat(np.array([M_u, M_d])) > 1.05  # informative failure
    assert split_rhat(np.array([E_u, E_d])) < 1.05  # E is blind to the sector
    assert split_rhat(np.array([np.abs(M_u), np.abs(M_d)])) < 1.05


@slow
@pytest.mark.parametrize("seed", SEEDS)
def test_D3_run_length_and_ess(seed):
    out = overdispersed(8, 3.5, N_RUN + BURN, seed)
    for which in ("M", "E"):
        x = _stack(out, which)
        taus = [tau1(c) for c in x]
        assert x.shape[1] >= 50 * max(taus), (which, taus)
        ess = x.size / np.mean(taus)
        assert ess >= 400, (which, ess)


@slow
@pytest.mark.parametrize("seed", SEEDS)
def test_D4_geweke_first_vs_second_half(seed):
    out = overdispersed(8, 3.5, N_RUN + BURN, seed)
    for name, (E, _) in out.items():
        x = E[BURN:]
        h = x.size // 2
        a, b = x[:h], x[h:]
        z = (a.mean() - b.mean()) / math.sqrt(se_mean(a) ** 2 + se_mean(b) ** 2)
        assert abs(z) < 3, (name, z)


def test_D_autocorr_time_on_AR1():
    """Sanity check of the estimator: AR(1) has tau = (1 + phi) / (1 - phi)."""
    rng = np.random.default_rng(3)
    phi, n = 0.9, 200_000
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = eps[0]
    for k in range(1, n):
        x[k] = phi * x[k - 1] + eps[k]
    assert tau1(x) == pytest.approx((1 + phi) / (1 - phi), rel=0.2)


# ---------------------------------------------------------------------------
# E. Injection and recovery
# ---------------------------------------------------------------------------


_B_ALL = bonds(MODEL4.states)
_B_VALS, _B_COUNTS = np.unique(_B_ALL, return_counts=True)
_B_LOGC = np.log(_B_COUNTS)


def test_E_log_partition_matches_brute_force():
    for r in [0.01, 0.2, 0.4, 1.0, 3.0, 6.0]:
        brute = logsumexp(r * _B_ALL)
        assert MODEL4.log_partition(r) == pytest.approx(brute, rel=1e-12)


class Quadrature:
    """Posterior on a (J, T) grid by direct summation (independent of module)."""

    def __init__(self, data, n_grid=300):
        b = bonds(data)
        K, sb = b.size, b.sum()
        Js = np.linspace(*J_BOUNDS, n_grid)
        Ts = np.linspace(*T_BOUNDS, n_grid)
        gJ, gT = np.meshgrid(Js, Ts, indexing="ij")
        self.r = gJ / gT
        logZ = logsumexp(self.r[..., None] * _B_VALS + _B_LOGC, axis=-1)
        lp = self.r * sb - K * logZ
        w = np.exp(lp - lp.max())
        self.w = w / w.sum()
        self.gJ, self.gT = gJ, gT
        order = np.argsort(self.r.ravel())
        self._rs = self.r.ravel()[order]
        self._cw = np.cumsum(self.w.ravel()[order])
        self.mean = float(np.sum(self.w * self.r))
        self.sd = float(np.sqrt(np.sum(self.w * self.r**2) - self.mean**2))

    def cdf(self, x):
        return np.interp(x, self._rs, self._cw, left=0.0, right=1.0)

    def quantile(self, q):
        return float(np.interp(q, self._cw, self._rs))


def info(arr):
    """arr: (steps, chains). Returns dict with mean, se, tau, thinned samples."""
    arr = np.asarray(arr, float)
    if arr.ndim == 1:
        arr = arr[:, None]
    t = tau_of(arr)
    flat = arr.ravel()
    return {
        "mean": flat.mean(),
        "sd": flat.std(),
        "tau": t,
        "n_eff": flat.size / t,
        "se": flat.std() / math.sqrt(flat.size / t),
        "thinned": arr[:: int(math.ceil(2 * t))].ravel(),
    }


def assert_matches_quadrature(r_arr, quad):
    s = info(r_arr)
    assert abs(s["mean"] - quad.mean) < 3 * s["se"], (s["mean"], quad.mean, s["se"])
    p = stats.kstest(s["thinned"], quad.cdf).pvalue
    assert p > 0.01, p


@slow
@pytest.mark.parametrize("seed", SEEDS)
def test_E1_mh_against_quadrature(seed):
    data = make_data(1000 + seed, K=20, J=1.0, T=2.5)
    post = ising.IsingPosterior(MODEL4, data)
    quad = Quadrature(data)
    chain, _ = run_mh(post, 1010 + seed)
    assert_matches_quadrature(chain[:, 0] / chain[:, 1], quad)


@slow
@pytest.mark.parametrize("seed", SEEDS[:1])
def test_E2_coverage_of_68_percent_interval(seed):
    rng = np.random.default_rng(1100 + seed)
    hits, n_rep = 0, 200
    for _ in range(n_rep):
        J = rng.uniform(*J_BOUNDS)
        T = rng.uniform(*T_BOUNDS)
        data = MODEL4.sample(J, T, size=20, rng=rng)
        post = ising.IsingPosterior(MODEL4, data)
        Js, Ts, w = post.grid_posterior(n_grid=300)
        r = (Js[:, None] / Ts[None, :]).ravel()
        order = np.argsort(r)
        cw = np.cumsum(w.ravel()[order])
        lo, hi = np.interp([0.16, 0.84], cw, r[order])
        hits += lo <= J / T <= hi
    assert 0.61 <= hits / n_rep <= 0.75, hits / n_rep


@slow
@pytest.mark.parametrize("seed", SEEDS[:1])
def test_E3_posterior_sd_shrinks_as_inverse_sqrt_K(seed):
    Ks = [5, 20, 80]
    sds = []
    for K in Ks:
        vals = [Quadrature(make_data(1200 + seed * 100 + k, K=K), n_grid=200).sd
                for k in range(20)]
        sds.append(np.mean(vals))
    slope = np.polyfit(np.log(Ks), np.log(sds), 1)[0]
    assert -0.65 < slope < -0.35, (slope, sds)


# ---------------------------------------------------------------------------
# F. Hand-written versus emcee
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference():
    emcee = pytest.importorskip("emcee")
    import time

    data = make_data(1300, K=20, J=1.0, T=2.5)
    post = ising.IsingPosterior(MODEL4, data)
    quad = Quadrature(data)

    t0 = time.perf_counter()
    chain, _ = run_mh(post, 1301)
    t_mh = time.perf_counter() - t0
    mh = {"r": (chain[:, 0] / chain[:, 1])[:, None], "T": chain[:, 1][:, None], "time": t_mh}

    rng = np.random.default_rng(1302)
    n_w, n_steps, n_burn = 32, 5000, 1000
    start = np.array([1.5, 3.0]) + 1e-2 * rng.standard_normal((n_w, 2))
    sampler = emcee.EnsembleSampler(n_w, 2, post)
    sampler.random_state = np.random.RandomState(1303).get_state()
    t0 = time.perf_counter()
    sampler.run_mcmc(start, n_burn + n_steps, progress=False)
    t_em = time.perf_counter() - t0
    wc = sampler.get_chain(discard=n_burn)
    em = {"r": wc[..., 0] / wc[..., 1], "T": wc[..., 1], "time": t_em}
    return {"quad": quad, "mh": mh, "emcee": em}


@slow
def test_F1_means_and_percentiles_agree(reference):
    for key in ("r", "T"):
        a, b = info(reference["mh"][key]), info(reference["emcee"][key])
        comb = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
        assert abs(a["mean"] - b["mean"]) < 3 * comb, (key, a["mean"], b["mean"], comb)
    qa = np.percentile(reference["mh"]["r"], [16, 50, 84])
    qb = np.percentile(reference["emcee"]["r"], [16, 50, 84])
    print("J/T 16/50/84  MH:", qa, " emcee:", qb)
    assert np.all(np.abs(qa - qb) < 0.5 * info(reference["mh"]["r"])["sd"])


@slow
def test_F2_two_sample_ks(reference):
    for key in ("r", "T"):
        a = info(reference["mh"][key])["thinned"]
        b = info(reference["emcee"][key])["thinned"]
        assert stats.ks_2samp(a, b).pvalue > 0.01, key


@slow
def test_F3_both_against_quadrature(reference):
    assert_matches_quadrature(reference["mh"]["r"], reference["quad"])
    assert_matches_quadrature(reference["emcee"]["r"], reference["quad"])


@slow
def test_F4_report_tau_and_ess_per_second(reference):
    """Descriptive: prints the numbers to discuss; only checks they are sane."""
    for name in ("mh", "emcee"):
        for key in ("r", "T"):
            s = info(reference[name][key])
            rate = s["n_eff"] / reference[name]["time"]
            print(f"{name:6s} {key}: tau = {s['tau']:8.1f}, ESS = {s['n_eff']:9.0f}, "
                  f"ESS/s = {rate:9.1f}")
            assert np.isfinite(rate) and rate > 0
