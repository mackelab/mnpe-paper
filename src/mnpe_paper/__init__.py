"""Shared machinery for the MNPE paper's experiments.

Three things live here rather than in any one experiment:

``calibration``
    Expected calibration error for discrete parameters, reliability diagrams,
    and the null-interval baselines an observed ECE must be judged against. This
    is the part with reuse value outside the paper, and it depends on nothing
    else in the package.

``paths`` and ``figure_io``
    Where results and artifacts live, and how a figure script chooses between a
    tracked summary and a full pipeline artifact.

``plotting``
    The paper's matplotlib style and its pairplot.

``models``
    One subpackage per experiment, holding its specification, prior, simulator,
    reference machinery and evaluation suite. Runnable scripts live outside the
    package, under ``experiments/``.

The method itself is not here. MNPE is implemented and maintained in the sbi
toolbox (https://sbi.readthedocs.io); this package holds only what reproduces
the paper's experiments.

Nothing is re-exported at this level, so importing the calibration utilities
does not pull in matplotlib or a repository-root lookup. The package version is
declared once, in ``pyproject.toml``, and is available through
``importlib.metadata.version("mnpe-paper")``.
"""
