"""The scalable Gaussian benchmark: D independent copies of the toy model.

Two sweeps share this model. One grows the number of discrete dimensions D, the
other the number of classes K per dimension. Because the model factorizes, the
exact posterior stays available in closed form at every size, so both sweeps are
scored against a known answer rather than against another sampler.
"""
