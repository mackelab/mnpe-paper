"""The Hodgkin-Huxley (jaxley) single-compartment neuron model.

Four continuous conductance and reversal parameters, plus two discrete channel
switches: a binary M-type potassium channel and a three-way calcium channel
choice (off, L-type, T-type). The simulator and training pipeline need jaxley and
live under ``experiments/hh/``; this package only reads their outputs.
"""
