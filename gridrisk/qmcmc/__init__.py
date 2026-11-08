"""Quantum-enhanced Markov chain Monte Carlo (Layden et al., Nature 2023), reused from our Project 1 code.

A quantum circuit proposes the next plan; a classical Metropolis step accepts or rejects it with the true
score, so the sampled distribution is exact even if the circuit is noisy.
"""
