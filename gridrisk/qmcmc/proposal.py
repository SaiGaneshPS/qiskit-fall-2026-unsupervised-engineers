"""Qiskit implementation of the quantum proposal step (Layden et al., Nature 619, 282, 2023).

From the current reserve design s, prepare |s>, evolve under
    H = (1 - gamma) * alpha * H_sur + gamma * sum_j X_j,
    H_sur = sum_j h_j Z_j + sum_{j<k} J_jk Z_j Z_k      (learned Ising surrogate)
for a random time t, and measure in the computational basis to get the proposal s'.

Trotterisation follows the paper's S15/S16: with H1 = single-qubit part and H2 = ZZ part,
    V~ = e^{-i H1 dt} (e^{-i H2 dt} e^{-i H1 dt})^(r-1),  dt = t / r,  r = ceil(t / DT).
V~ is a palindrome of complex-symmetric factors, so V~^T = V~ and the proposal
Q(s'|s) = |<s'|V~|s>|^2 is exactly symmetric, as Metropolis-Hastings requires.

SPAM twirling (paper Sec. S2): pick a random key c, prepare s XOR c, flip the signs
h_j -> c_j h_j and J_jk -> c_j c_k J_jk, and XOR the measured string with c. Ideal
statistics are unchanged; biased readout/T1 errors are symmetrised.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Operator

from .mcmc import mixer_alpha

DT = 0.8                      # Trotter step used in the paper's hardware runs
GAMMA_RANGE = (0.25, 0.60)    # paper's randomised parameter ranges
TIME_RANGE = (2.0, 20.0)


def draw_parameters(rng: np.random.Generator, k: int) -> tuple[np.ndarray, np.ndarray]:
    return rng.uniform(*GAMMA_RANGE, k), rng.uniform(*TIME_RANGE, k)


def n_trotter_steps(t: float, dt: float = DT) -> int:
    return max(1, math.ceil(t / dt - 1e-12))


def _u3_params(a: float, b: float) -> tuple[float, float, float]:
    """(theta, phi, lam) of U3 equal (up to phase) to exp(-i (a X + b Z))."""
    w = math.hypot(a, b)
    if w < 1e-15:
        return 0.0, 0.0, 0.0
    c, s = math.cos(w), math.sin(w)
    M = np.array([[c - 1j * s * b / w, -1j * s * a / w],
                  [-1j * s * a / w, c + 1j * s * b / w]])
    # U3(theta, phi, lam) = [[cos(th/2), -e^{i lam} sin(th/2)], [e^{i phi} sin(th/2), e^{i(phi+lam)} cos(th/2)]]
    theta = 2 * math.atan2(abs(M[1, 0]), abs(M[0, 0]))
    ph0 = np.angle(M[0, 0]) if abs(M[0, 0]) > 1e-12 else 0.0
    phi = np.angle(M[1, 0]) - ph0 if abs(M[1, 0]) > 1e-12 else 0.0
    lam = np.angle(-M[0, 1]) - ph0 if abs(M[0, 1]) > 1e-12 else np.angle(M[1, 1]) - ph0 - phi
    return float(theta), float(phi), float(lam)


@dataclass
class ProposalTemplate:
    """Parameterised circuit for a fixed number of Trotter steps r."""
    circuit: QuantumCircuit
    r: int
    pairs: list[tuple[int, int]]
    init: ParameterVector
    u: ParameterVector
    zz: ParameterVector


def build_template(n: int, pairs: list[tuple[int, int]], r: int, measure: bool = True) -> ProposalTemplate:
    init = ParameterVector("b", n)            # RX(pi * b_j) prepares bit b_j
    u = ParameterVector("u", 3 * n)           # U3 angles of e^{-i H1 dt} on each qubit
    zz = ParameterVector("z", len(pairs))     # RZZ angles of e^{-i H2 dt}
    qc = QuantumCircuit(n, n if measure else 0)
    for j in range(n):
        qc.rx(math.pi * init[j], j)

    def h1_layer():
        for j in range(n):
            qc.u(u[3 * j], u[3 * j + 1], u[3 * j + 2], j)

    h1_layer()
    for _ in range(r - 1):
        for e, (j, k) in enumerate(pairs):
            qc.rzz(zz[e], j, k)
        h1_layer()
    if measure:
        qc.measure(range(n), range(n))
    return ProposalTemplate(qc, r, list(pairs), init, u, zz)


def bind_values(tpl: ProposalTemplate, h: np.ndarray, J: np.ndarray, gamma: float, t: float,
                bits: np.ndarray, key: np.ndarray | None = None) -> np.ndarray:
    """Parameter vector for one proposal from configuration `bits` (0/1 array).

    `key` (0/1 array) applies SPAM twirling: prepare bits ^ key with sign-flipped
    couplings; the caller XORs the measured string with the same key.
    """
    n = len(h)
    alpha = mixer_alpha(h, J)
    dt = t / tpl.r
    sgn = np.ones(n) if key is None else 1 - 2 * np.asarray(key, float)
    prep = np.asarray(bits) if key is None else np.asarray(bits) ^ np.asarray(key)
    vals_u = []
    for j in range(n):
        vals_u.extend(_u3_params(gamma * dt, (1 - gamma) * alpha * h[j] * sgn[j] * dt))
    vals_zz = [2 * (1 - gamma) * alpha * J[j, k] * sgn[j] * sgn[k] * dt for j, k in tpl.pairs]
    return np.concatenate([prep.astype(float), vals_u, vals_zz])


def ordered_parameters(tpl: ProposalTemplate) -> list:
    return list(tpl.init) + list(tpl.u) + list(tpl.zz)


# --------------------------------------------------------------------------- native (rz, sx, cz) form
# Qiskit's basis translator slows down sharply on parameterised U/RZZ chains, so the
# hardware templates are written directly in IBM Heron's native gates. Parameters only
# appear as RZ angles. Identities used (up to global phase):
#   U3(th, ph, la) = RZ(ph + pi) SX RZ(th + pi) SX RZ(la)       (circuit order: right to left)
#   RZZ(th)_jk     = H_k CZ RX_k(th) CZ H_k,  RX(th) = U3(th, -pi/2, pi/2)

def _native_u3(qc, q, th, ph, la):
    qc.rz(la, q)
    qc.sx(q)
    qc.rz(th + math.pi, q)
    qc.sx(q)
    qc.rz(ph + math.pi, q)


def _native_h(qc, q):
    _native_u3(qc, q, math.pi / 2, 0.0, math.pi)


@dataclass
class NativeTemplate:
    circuit: QuantumCircuit
    r: int
    pairs: list[tuple[int, int]]
    first: ParameterVector     # U3 angles of (e^{-i H1 dt} X^{prep}) per qubit
    mid: ParameterVector       # U3 angles of e^{-i H1 dt} per qubit (shared by later layers)
    zz: ParameterVector        # RZZ angles

    @property
    def parameters(self):
        return list(self.first) + list(self.mid) + list(self.zz)


def build_native_template(n: int, pairs: list[tuple[int, int]], r: int) -> NativeTemplate:
    first, mid, zz = ParameterVector("f", 3 * n), ParameterVector("m", 3 * n), ParameterVector("z", len(pairs))
    qc = QuantumCircuit(n, n)
    for j in range(n):
        _native_u3(qc, j, first[3 * j], first[3 * j + 1], first[3 * j + 2])
    for _ in range(r - 1):
        for e, (j, k) in enumerate(pairs):
            _native_h(qc, k)
            qc.cz(j, k)
            _native_u3(qc, k, zz[e], -math.pi / 2, math.pi / 2)
            qc.cz(j, k)
            _native_h(qc, k)
        for j in range(n):
            _native_u3(qc, j, mid[3 * j], mid[3 * j + 1], mid[3 * j + 2])
    qc.measure(range(n), range(n))
    return NativeTemplate(qc, r, list(pairs), first, mid, zz)


def bind_native(tpl: NativeTemplate, h, J, gamma, t, bits, key=None) -> np.ndarray:
    """Parameter values in tpl.parameters order (same physics as bind_values)."""
    from qiskit.synthesis import OneQubitEulerDecomposer
    dec = OneQubitEulerDecomposer("U3")
    n = len(h)
    alpha = mixer_alpha(h, J)
    dt = t / tpl.r
    sgn = np.ones(n) if key is None else 1 - 2 * np.asarray(key, float)
    prep = np.asarray(bits) if key is None else np.asarray(bits) ^ np.asarray(key)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    f_vals, m_vals = [], []
    for j in range(n):
        a, b = gamma * dt, (1 - gamma) * alpha * h[j] * sgn[j] * dt
        w = math.hypot(a, b)
        c, s = math.cos(w), (math.sin(w) / w if w > 1e-15 else 0.0)
        M = np.array([[c - 1j * s * b, -1j * s * a], [-1j * s * a, c + 1j * s * b]])
        m_vals.extend(dec.angles(M))
        f_vals.extend(dec.angles(M @ X if prep[j] else M))
    zz_vals = [2 * (1 - gamma) * alpha * J[j, k] * sgn[j] * sgn[k] * dt for j, k in tpl.pairs]
    return np.array(f_vals + m_vals + zz_vals)


def coupling_pairs(J: np.ndarray, tol: float = 1e-12) -> list[tuple[int, int]]:
    n = J.shape[0]
    return [(i, j) for i in range(n) for j in range(i + 1, n) if abs(J[i, j]) > tol]


def trotter_proposal_matrix(h, J, gammas, times, pairs=None) -> np.ndarray:
    """Exact proposal matrix of the Trotterised circuits (noiseless), for small n."""
    n = len(h)
    pairs = coupling_pairs(J) if pairs is None else pairs
    N = 2 ** n
    Q = np.zeros((N, N))
    cache: dict[int, ProposalTemplate] = {}
    for g, t in zip(gammas, times):
        r = n_trotter_steps(t)
        if r not in cache:
            cache[r] = build_template(n, pairs, r, measure=False)
        tpl = cache[r]
        vals = bind_values(tpl, h, J, g, t, np.zeros(n, dtype=int))
        qc = tpl.circuit.assign_parameters(dict(zip(ordered_parameters(tpl), vals)))
        # the RX(pi b) prefix is identity at b = 0, so this is the unitary V~ itself
        U = Operator(qc).data
        Q += np.abs(U.T) ** 2
    return Q / len(gammas)
