"""Exercise an installed distribution without importing from the checkout.

Run with the target environment's ``python -I scripts/smoke_install.py``.
CI and release-package tests use this same generation/persistence path.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import prior_generator as pg


def main() -> None:
    """Generate, reconstruct, persist, and validate a small real corpus."""
    config = pg.make_scm_prior(
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        n_time_steps=24,
        n_cells=2,
        draws_per_cell=2,
        seed=731,
    )
    world = pg.sample_scm(config, seed=731)
    np.testing.assert_allclose(world.reconstruction(), world.data["sales"], rtol=1e-12, atol=1e-12)
    corpus = pg.sample_prior_predictive(config)
    with TemporaryDirectory(prefix="prior-generator-smoke-") as directory:
        path = Path(directory) / "corpus.npz"
        pg.save_corpus(corpus, path)
        loaded = pg.load_corpus(path)
        errors = pg.DataGenerator.validate_corpus(loaded)
        if errors:
            raise AssertionError("Invalid persisted corpus:\n" + "\n".join(errors))
        np.testing.assert_array_equal(loaded["spend_raw"], corpus["spend_raw"])
    print(f"prior-generator {pg.__version__}: generation, decomposition, and persistence pass")


if __name__ == "__main__":
    main()
