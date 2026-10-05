"""Per-experiment specifications, priors, simulators and evaluation suites.

One subpackage per experiment. The split between this package and
``experiments/`` is by run entry point: anything importable — a specification, a
prior, a simulator, a reference implementation, an evaluation suite — lives here;
anything with a ``__main__`` block lives under ``experiments/``.

Some of these modules must keep a stable import path for reasons beyond
tidiness: cached checkpoints resolve their estimator classes by module name, so
moving such a class renames it out of every artifact that references it.
"""
