"""Quantum loss-of-load estimator: every combination of cold-snap demand and plant outages in one state.

Structure (adapted from quantum credit-risk analysis, Woerner and Egger 2019; Egger et al. 2021):
power plants play the role of loans that can "default" (be forced out), demand plays the role of the
loss threshold, and the latent factor that correlates defaults is the weather.

    demand register  |b>        n_d qubits, amplitudes sqrt(P(demand level b)) from the AI demand model
    unit qubits      |f_k>      1 = unit k forced out; P(f_k = 1 | b) via a uniformly controlled RY,
                                so outage risk can rise in the coldest (highest-demand) hours
    sum register     |s>        s = sum_k c_k f_k + step * b   (quantum weighted adder, integer MW quanta)
    flag             |s >= T>   T = (total capacity - base demand) / quantum + 1  (integer comparator)

so  A|0> = sum_{b,f} sqrt(P(b) P(f | b)) |b, f, s(b, f), [s >= T]>  and  P(flag = 1) = loss-of-load probability.

Amplitude estimation then reads that probability with error O(1/N) in N uses of A, versus O(1/sqrt(N))
for Monte Carlo. Every gate count is polynomial in the number of units (adder: O(K log W)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import IntegerComparatorGate, StatePreparation, UCRYGate, WeightedSumGate

BASIS = ["cx", "u", "x", "h", "ry", "z"]


def _angle(p: float) -> float:
    return 2.0 * math.asin(math.sqrt(min(max(p, 0.0), 1.0)))


@dataclass
class LolpInstance:
    """A discretised loss-of-load question: is total available capacity below demand?

    Capacities and demand are integers in units of `quantum_mw`. Demand = base + step * b for level b.
    `p_fail[k, b, s]` is the forced-outage probability of unit k at demand level b when its parent unit
    (`parent[k]`, an earlier unit, or None) is in state s (1 = out). This captures common-mode failures
    such as both poles of one HVDC line, or a whole hydro plant losing its intake to frazil ice in a cold snap.
    """
    names: list[str]
    cap: list[int]                    # unit capacities (quanta)
    p_fail: np.ndarray                # (K, 2**n_d, 2)
    p_level: np.ndarray               # (2**n_d,) demand-level probabilities
    base: int                         # demand at level 0 (quanta), net of firm (non-modelled) supply
    step: int                         # demand increment per level (quanta)
    parent: list = field(default_factory=list)
    quantum_mw: float = 50.0
    labels: dict = field(default_factory=dict)

    def __post_init__(self):
        self.p_fail = np.asarray(self.p_fail, float)
        if self.p_fail.ndim == 2:
            self.p_fail = np.repeat(self.p_fail[:, :, None], 2, axis=2)
        if not self.parent:
            self.parent = [None] * len(self.cap)
        self.p_level = np.asarray(self.p_level, float) / np.sum(self.p_level)

    @property
    def K(self) -> int:
        return len(self.cap)

    @property
    def n_d(self) -> int:
        return int(round(math.log2(len(self.p_level))))

    @property
    def threshold(self) -> int:
        """Shortfall iff lost capacity + step*b >= T  (i.e. demand strictly above available capacity)."""
        return sum(self.cap) - self.base + 1

    def weights(self) -> list[int]:
        return list(self.cap) + [self.step * 2 ** j for j in range(self.n_d)]

    # --------------------------------------------------------------------- classical ground truth
    def _pattern_probs(self, b: int) -> tuple[np.ndarray, np.ndarray]:
        """All 2^K outage patterns and their probabilities at demand level b (respecting parents)."""
        pats = ((np.arange(2 ** self.K)[:, None] >> np.arange(self.K)) & 1).astype(np.int8)
        pr = np.ones(len(pats))
        for k in range(self.K):
            s = pats[:, self.parent[k]] if self.parent[k] is not None else np.zeros(len(pats), np.int8)
            q = self.p_fail[k, b, s]
            pr *= np.where(pats[:, k] == 1, q, 1 - q)
        return pats, pr

    def exact(self) -> float:
        """Exact LOLP by enumerating every outage pattern at every demand level."""
        p = 0.0
        caps = np.asarray(self.cap)
        for b, pb in enumerate(self.p_level):
            if pb == 0:
                continue
            pats, pr = self._pattern_probs(b)
            short = pats @ caps + self.step * b >= self.threshold
            p += pb * pr[short].sum()
        return float(p)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Monte Carlo: n independent (demand, outage) draws; returns 0/1 shortfall indicators."""
        b = rng.choice(len(self.p_level), size=n, p=self.p_level)
        f = np.zeros((n, self.K), np.int8)
        for k in range(self.K):
            s = f[:, self.parent[k]] if self.parent[k] is not None else np.zeros(n, np.int8)
            f[:, k] = rng.random(n) < self.p_fail[k, b, s]
        lost = f @ np.asarray(self.cap)
        return (lost + self.step * b >= self.threshold).astype(np.int8)

    def importance_sample(self, n: int, rng: np.random.Generator, tilt: float = 1.0, lam: float = 0.0) -> np.ndarray:
        """Classical importance sampling: outage probabilities raised to q = 1 - (1 - p)^tilt and demand levels
        tilted by exp(lam * b); returns weighted shortfall indicators (their mean is the LOLP)."""
        L = len(self.p_level)
        ql = self.p_level * np.exp(lam * np.arange(L))
        ql /= ql.sum()
        b = rng.choice(L, size=n, p=ql)
        w = self.p_level[b] / ql[b]
        f = np.zeros((n, self.K), np.int8)
        for k in range(self.K):
            s = f[:, self.parent[k]] if self.parent[k] is not None else np.zeros(n, np.int8)
            p = self.p_fail[k, b, s]
            q = np.where(p > 0, 1 - (1 - p) ** tilt, 0.0)
            x = rng.random(n) < q
            w *= np.where(x, np.divide(p, q, out=np.ones_like(p), where=q > 0),
                          np.divide(1 - p, 1 - q, out=np.ones_like(p), where=q < 1))
            f[:, k] = x
        short = f @ np.asarray(self.cap) + self.step * b >= self.threshold
        return w * short


# ------------------------------------------------------------------------------------------- circuits
def state_preparation(inst: LolpInstance, flat: bool = True) -> tuple[QuantumCircuit, int]:
    """Build A. Returns (circuit, index of the flag qubit)."""
    K, nd = inst.K, inst.n_d
    wsum = WeightedSumGate(K + nd, inst.weights())
    ns = wsum.num_qubits - (K + nd)
    units = QuantumRegister(K, "unit")
    dem = QuantumRegister(nd, "demand")
    s = QuantumRegister(ns, "sum")
    flag = QuantumRegister(1, "shortfall")
    anc = AncillaRegister(max(ns, 2), "anc")
    qc = QuantumCircuit(dem, units, s, flag, anc, name="A_grid")
    qc.append(StatePreparation(np.sqrt(inst.p_level)), dem)
    L = 2 ** nd
    for k in range(K):
        par = inst.parent[k]
        # UCRY convention: control value = sum_j bit_j 2^j over the controls in order
        # (demand bits first, then the parent unit's outage bit as the most significant control)
        angles = [_angle(inst.p_fail[k, v % L, v // L]) for v in range(L * (2 if par is not None else 1))]
        if np.allclose(angles, angles[0]):
            qc.ry(angles[0], units[k])
        elif par is None and np.allclose(inst.p_fail[k, :, 0], inst.p_fail[k, 0, 0]):
            qc.ry(angles[0], units[k])
        else:
            ctrls = list(dem) + ([units[par]] if par is not None else [])
            qc.append(UCRYGate(angles), [units[k]] + ctrls)
    qc.append(wsum, list(units) + list(dem) + list(s))
    qc.append(IntegerComparatorGate(ns, inst.threshold, geq=True), list(s) + [flag[0]])
    fidx = qc.find_bit(flag[0]).index
    if flat:
        qc = transpile(qc, basis_gates=BASIS, optimization_level=1, qubits_initially_zero=True)
        qc.name = "A_grid"
    return qc, fidx


def statevector_probability(A: QuantumCircuit, flag: int) -> float:
    """Exact P(flag = 1) of the circuit, from Aer's statevector simulator."""
    from qiskit_aer import AerSimulator
    qc = A.copy()
    qc.save_probabilities_dict(qubits=[flag])
    sim = AerSimulator(method="statevector")
    d = sim.run(transpile(qc, sim, optimization_level=0), shots=1).result().data(0)["probabilities"]
    return float(sum(v for k, v in d.items() if int(str(k), 0) == 1))


def grover_operator(A: QuantumCircuit, flag: int) -> QuantumCircuit:
    """Q = A S0 A^dagger S_chi with S_chi a Z on the flag qubit."""
    from qiskit.circuit.library import grover_operator as qiskit_grover
    oracle = QuantumCircuit(A.num_qubits)
    oracle.z(flag)
    return qiskit_grover(oracle, state_preparation=A)


def mlae_circuits(A: QuantumCircuit, flag: int, powers: list[int]) -> list[QuantumCircuit]:
    """Circuits Q^m A |0> with only the flag measured (maximum-likelihood AE, Suzuki et al. 2020)."""
    Q = transpile(grover_operator(A, flag), basis_gates=BASIS, optimization_level=1)
    out = []
    for m in powers:
        qc = QuantumCircuit(A.num_qubits, 1)
        qc.compose(A, inplace=True)
        for _ in range(m):
            qc.compose(Q, inplace=True)
        qc.measure(flag, 0)
        out.append(qc)
    return out


def mlae_estimate(hits, shots, powers, grid: int = 20001) -> float:
    """Maximum-likelihood a from counts of flag=1 after Q^m A: P(1) = sin^2((2m+1) theta)."""
    theta = np.linspace(0, np.pi / 2, grid)
    ll = np.zeros_like(theta)
    for h, s, m in zip(hits, shots, powers):
        p = np.clip(np.sin((2 * m + 1) * theta) ** 2, 1e-15, 1 - 1e-15)
        ll += h * np.log(p) + (s - h) * np.log(1 - p)
    return float(np.sin(theta[np.argmax(ll)]) ** 2)


def mlae_estimate_fine(hits, shots, powers) -> float:
    """MLAE with a coarse grid then a local refinement (needed for very small probabilities)."""
    theta = np.linspace(0, np.pi / 2, 20001)

    def ll(th):
        out = np.zeros_like(th)
        for h, s, m in zip(hits, shots, powers):
            p = np.clip(np.sin((2 * m + 1) * th) ** 2, 1e-300, 1 - 1e-16)
            out += h * np.log(p) + (s - h) * np.log1p(-p)
        return out

    t0 = theta[np.argmax(ll(theta))]
    d = theta[1] - theta[0]
    fine = np.linspace(max(t0 - d, 0), min(t0 + d, np.pi / 2), 4001)
    return float(np.sin(fine[np.argmax(ll(fine))]) ** 2)


def oracle_queries(powers, shots) -> int:
    """Applications of A counted as in the AE literature: each shot of Q^m A uses 2m + 1 queries."""
    return int(sum(s * (2 * m + 1) for m, s in zip(powers, shots)))


def noise_aware_mlae(hits, shots, powers, n_theta=4001, gammas=np.linspace(0.0, 0.6, 121)):
    """Joint ML fit of theta and a depolarising decay rate g (Tanaka et al. 2021); returns (a_hat, g_hat).

    P_obs(m) = exp(-g (2m+1)) sin^2((2m+1) theta) + (1 - exp(-g (2m+1))) / 2
    """
    th = np.linspace(0, np.pi / 2, n_theta)[:, None]
    g = gammas[None, :]
    ll = np.zeros((len(th), len(gammas)))
    for h, s, m in zip(hits, shots, powers):
        d = np.exp(-g * (2 * m + 1))
        p = np.clip(d * np.sin((2 * m + 1) * th) ** 2 + (1 - d) / 2, 1e-12, 1 - 1e-12)
        ll += h * np.log(p) + (s - h) * np.log(1 - p)
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    return float(np.sin(th[i, 0]) ** 2), float(gammas[j])


def resource_counts(A: QuantumCircuit, flag: int) -> dict:
    """Qubits and two-qubit gate counts of A and of one Grover iterate Q (all-to-all basis)."""
    Q = transpile(grover_operator(A, flag), basis_gates=BASIS, optimization_level=1)
    return dict(qubits=A.num_qubits, cx_A=int(A.count_ops().get("cx", 0)), cx_Q=int(Q.count_ops().get("cx", 0)),
                depth_A=A.depth(), depth_Q=Q.depth())


# ------------------------------------------------------------------------------- hardware-scale circuit
def three_qubit_reduction(inst: LolpInstance, unit: int, level_split: int) -> dict:
    """Exact 3-qubit reduction of an instance for real hardware.

    q0 = "cold" (demand level >= level_split), q1 = unit `unit` forced out, q2 = shortfall.
    P(q2 = 1 | q0, q1) is the probability that the *rest* of the fleet cannot cover the demand,
    computed exactly by classical convolution, so P(q2 = 1) equals the full instance's LOLP.
    """
    pl = inst.p_level
    cold = np.arange(len(pl)) >= level_split
    p_cold = float(pl[cold].sum())
    caps = np.asarray(inst.cap)
    joint = np.zeros((2, 2))          # P(cold, unit out)
    short = np.zeros((2, 2))          # P(cold, unit out, shortfall)
    for b, pb in enumerate(pl):
        pats, pr = inst._pattern_probs(b)
        s = pats @ caps + inst.step * b >= inst.threshold
        c_ = int(cold[b])
        for o in (0, 1):
            m = pats[:, unit] == o
            joint[c_, o] += pb * pr[m].sum()
            short[c_, o] += pb * pr[m & s].sum()
    p_out = {c_: float(joint[c_, 1] / joint[c_].sum()) for c_ in (0, 1)}
    cond = {(c_, o): float(short[c_, o] / joint[c_, o]) if joint[c_, o] > 0 else 0.0 for c_ in (0, 1) for o in (0, 1)}
    return dict(p_cold=p_cold, p_out=p_out, cond=cond, p=float(short.sum()))


def load_reduction(d: dict) -> dict:
    """A reduction as saved in a results file (JSON keys are strings)."""
    return dict(p_cold=d["p_cold"], p_out={int(k): v for k, v in d["p_out"].items()},
                cond={tuple(int(c) for c in k.strip("()").split(",")): v for k, v in d["cond"].items()}, p=d.get("p"))


def three_qubit_A(red: dict) -> QuantumCircuit:
    qc = QuantumCircuit(3, name="A3")
    qc.ry(_angle(red["p_cold"]), 0)
    qc.append(UCRYGate([_angle(red["p_out"][0]), _angle(red["p_out"][1])]), [1, 0])
    # control value k = q0 + 2 q1
    qc.append(UCRYGate([_angle(red["cond"][(k & 1, (k >> 1) & 1)]) for k in range(4)]), [2, 0, 1])
    return qc


def three_qubit_circuits(red: dict, powers) -> list[QuantumCircuit]:
    A = three_qubit_A(red)
    Q = QuantumCircuit(3, name="Q")
    Q.z(2)                                                   # S_chi: mark "shortfall"
    Q.compose(A.inverse(), inplace=True)
    Q.x([0, 1, 2]); Q.h(2); Q.ccx(0, 1, 2); Q.h(2); Q.x([0, 1, 2])   # S_0 up to a global phase
    Q.compose(A, inplace=True)
    out = []
    for m in powers:
        qc = QuantumCircuit(3, 1)
        qc.compose(A, inplace=True)
        for _ in range(m):
            qc.compose(Q, inplace=True)
        qc.measure(2, 0)
        out.append(qc)
    return out
