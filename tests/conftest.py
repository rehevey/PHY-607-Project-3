"""Shared test setup.

* Registers the ``slow`` marker (run fast tests only with ``pytest -m "not slow"``).
* Loads the implementation module. The file name starts with a digit, so it
  cannot be imported normally. Set ``ISING_MODULE`` to override the path.
* ``SEEDS``: seeds used by the slow tests. Raise with ``ISING_NSEEDS=10``.
"""

import importlib.util
import os
import pathlib

os.environ.setdefault("MPLBACKEND", "Agg")

SEEDS = list(range(int(os.environ.get("ISING_NSEEDS", "3"))))

_CANDIDATES = ["2Disling_V2.py", "ising_inference_v2.py"]


def load_ising():
    path = os.environ.get("ISING_MODULE")
    if path is None:
        here = pathlib.Path(__file__).resolve().parent
        for folder in (here.parent, here):
            for name in _CANDIDATES:
                if (folder / name).exists():
                    path = str(folder / name)
                    break
            if path:
                break
    if path is None:
        raise FileNotFoundError("Set ISING_MODULE to the implementation file.")
    spec = importlib.util.spec_from_file_location("ising_impl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running statistical test")
