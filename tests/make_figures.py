import numpy as np
import matplotlib.pyplot as plt
from conftest import load_ising

ising = load_ising()

def run_M(init, T, n_sweeps, rng, n_burn=1000, J=1.0):
    s = init.copy()
    for _ in range(n_burn):
        ising.metropolis_sweep(s, J, T, rng)
    M = np.empty(n_sweeps)
    for k in range(n_sweeps):
        ising.metropolis_sweep(s, J, T, rng)
        M[k] = s.mean()
    return M

n = 4
rng = np.random.default_rng(600)
edges = -1.0625 + 0.125 * np.arange(18)   # M takes values in steps of 2/16

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
for ax, T in zip(axes, (1.5, 3.0)):
    up = run_M(np.ones((n, n), np.int8), T, 40_000, rng)
    dn = run_M(-np.ones((n, n), np.int8), T, 40_000, rng)
    ax.hist(np.concatenate([up, dn]), bins=edges, density=True,
            alpha=0.6, label="pooled")
    ax.hist(up, bins=edges, density=True, histtype="step", label="up start")
    ax.hist(dn, bins=edges, density=True, histtype="step", label="down start")
    ax.set_title(f"4x4, T = {T}")
    ax.set_xlabel("M per spin")
axes[0].set_ylabel("density")
axes[0].legend()
fig.tight_layout()
fig.savefig("M_histogram.png", dpi=150)

import matplotlib.pyplot as plt

BURN = 500
N_RUN = 5000
_cache = {}

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

def overdispersed(n, T, n_run, seed):
    key = (n, T, n_run, seed)
    if key not in _cache:
        rng = np.random.default_rng(900 + seed)
        out = {}
        for name, init in starts(n, rng).items():
            out[name] = run_chain(init, 1.0, T, n_run, rng, n_burn=0)
        _cache[key] = out
    return _cache[key]

import os

def test_D1_trace_plots(seed):
    import pathlib
    import matplotlib.pyplot as plt

    out = overdispersed(8, 3.5, N_RUN + BURN, seed)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    for name, (E, M) in out.items():
        ax[0].plot(E[:2000], lw=0.6, label=name)
        ax[1].plot(np.abs(M[:2000]), lw=0.6, label=name)
    ax[0].set_ylabel("E/N^2")
    ax[1].set_ylabel("|M|")
    ax[0].set_xlabel("sweep")
    ax[1].set_xlabel("sweep")
    ax[0].legend()
    fig.tight_layout()
    path = pathlib.Path("traces.png").resolve()
    fig.savefig(path, dpi=150)
    print("saved to", path)
    assert path.exists()

from test_ising import starts, run_chain  # same helpers the test uses

rng = np.random.default_rng(900)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for name, init in starts(8, rng).items():
    E, M = run_chain(init, 1.0, 3.5, 2000, rng, n_burn=0)
    ax[0].plot(E, lw=0.6, label=name)
    ax[1].plot(np.abs(M), lw=0.6, label=name)
ax[0].set_ylabel("E/N^2"); ax[1].set_ylabel("|M|")
ax[0].set_xlabel("sweep"); ax[1].set_xlabel("sweep")
ax[0].legend()
fig.tight_layout()
fig.savefig("traces.png", dpi=150)
print("saved to", os.path.abspath("traces.png"))
