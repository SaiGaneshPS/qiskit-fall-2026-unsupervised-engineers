"""Chronological extension: will the grid get through a multi-day cold snap?

Single-hour adequacy (gridrisk.instances) treats every hour independently, which is what makes the exact classical
convolution fast. Real cold snaps last days, and big outages persist: the January 2026 frazil shutdown kept
Bay d'Espoir off for about four days; a Labrador-Island Link (LIL) outage from ice can last days to weeks. Then each
unit's state must be tracked across time, the exact state space grows like 2^(units) per step, and planning tools
fall back on chronological Monte Carlo.

Model (one register per ingredient):
    severity |s>      n_s qubits: how severe the cold snap is (classes of the worst 3-day window of each of 46
                      winters, from the AI demand model)
    unit-days |f_kt>  1 = big unit k out on day t; day 1: P(out | s); later days: Markov persistence
                      P(out_t | out_{t-1}) = stay_k, P(out_t | in_{t-1}, s) = new_k(s)   (controlled rotations)
    per day t         weighted adder: lost capacity + net load Y_t(s) -> comparator -> flag_t; adder uncomputed
    final flag        OR of the daily flags = "at least one day with a shortfall"
Gate count grows linearly with units x days; exact enumeration grows like 2^(units x days).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import IntegerComparatorGate, StatePreparation, UCRYGate, WeightedSumGate

from .quantum import BASIS, _angle


@dataclass
class ColdSnapInstance:
    names: list[str]
    cap: list[int]               # big-unit capacities (quanta)
    p_sev: np.ndarray            # (S,) severity-class probabilities
    first: np.ndarray            # (K, S) P(out on day 1 | s)
    stay: np.ndarray             # (K,)   P(out_t | out_{t-1})
    new: np.ndarray              # (K, S) P(out_t | in_{t-1}, s)
    y0: np.ndarray               # (T,)   net load at severity 0 minus big capacity (quanta)
    w_sev: np.ndarray            # (T, n_s) non-negative weights of the severity bits in each day's net load
    quantum_mw: float = 50.0
    labels: dict = field(default_factory=dict)

    @property
    def K(self):
        return len(self.cap)

    @property
    def T(self):
        return len(self.y0)

    @property
    def n_s(self):
        return int(round(math.log2(len(self.p_sev))))

    def y(self, s: int) -> np.ndarray:
        bits = np.array([(s >> j) & 1 for j in range(self.n_s)])
        return self.y0 + self.w_sev @ bits

    # ------------------------------------------------------------------ classical
    def exact(self, persistence: bool = True) -> float:
        """Exact P(any shortfall day) by enumerating all unit trajectories (2^(K T) per severity class)."""
        K, T = self.K, self.T
        n = K * T
        pats = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).reshape(-1, K, T).astype(np.int8)
        caps = np.asarray(self.cap)
        p = 0.0
        for s, ps in enumerate(self.p_sev):
            pr = np.ones(len(pats))
            for k in range(K):
                stat = self.first[k, s]
                for t in range(T):
                    x = pats[:, k, t]
                    if t == 0 or not persistence:
                        q = np.full(len(pats), stat)
                    else:
                        q = np.where(pats[:, k, t - 1] == 1, self.stay[k], self.new[k, s])
                    pr *= np.where(x == 1, q, 1 - q)
            lost = np.einsum("nkt,k->nt", pats, caps)
            short = (lost + self.y(s)[None, :] >= 1).any(axis=1)
            p += ps * pr[short].sum()
        return float(p)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        s = rng.choice(len(self.p_sev), size=n, p=self.p_sev)
        caps = np.asarray(self.cap)
        state = np.zeros((n, self.K), np.int8)
        short = np.zeros(n, bool)
        ys = np.array([self.y(v) for v in range(len(self.p_sev))])
        for t in range(self.T):
            for k in range(self.K):
                q = self.first[k, s] if t == 0 else np.where(state[:, k] == 1, self.stay[k], self.new[k, s])
                state[:, k] = rng.random(n) < q
            short |= state @ caps + ys[s, t] >= 1
        return short.astype(np.int8)


def circuit(inst: ColdSnapInstance, flat: bool = True) -> tuple[QuantumCircuit, int]:
    K, T, ns = inst.K, inst.T, inst.n_s
    S = 2 ** ns
    sev = QuantumRegister(ns, "severity")
    units = [QuantumRegister(T, f"u{k}_{''.join(ch for ch in inst.names[k][:12] if ch.isalnum())}") for k in range(K)]
    weights_t = [list(inst.cap) + [int(v) for v in inst.w_sev[t]] for t in range(T)]
    n_sum = max(WeightedSumGate(K + ns, w).num_qubits - (K + ns) for w in weights_t)
    s_reg = QuantumRegister(n_sum, "sum")
    flags = QuantumRegister(T, "day_short")
    final = QuantumRegister(1, "any_short")
    anc = AncillaRegister(max(n_sum, 2), "anc")
    qc = QuantumCircuit(sev, *units, s_reg, flags, final, anc, name="A_coldsnap")
    qc.append(StatePreparation(np.sqrt(inst.p_sev)), sev)
    for k in range(K):
        for t in range(T):
            if t == 0:
                ang = [_angle(inst.first[k, s]) for s in range(S)]
                qc.append(UCRYGate(ang), [units[k][0]] + list(sev))
            else:
                # control value = s + S * (unit k out on day t-1)
                ang = [_angle(inst.new[k, v % S] if v < S else inst.stay[k]) for v in range(2 * S)]
                qc.append(UCRYGate(ang), [units[k][t]] + list(sev) + [units[k][t - 1]])
    for t in range(T):
        w = weights_t[t]
        g = WeightedSumGate(K + ns, w)
        nq = g.num_qubits - (K + ns)
        # synthesise the adder once (with clean ancillas) so that its exact inverse can uncompute the sum register
        sub = QuantumCircuit(K + ns + nq + len(anc))
        sub.append(g, range(K + ns + nq))
        sub = transpile(sub, basis_gates=BASIS, optimization_level=1, qubits_initially_zero=True)
        ins = [units[k][t] for k in range(K)] + list(sev)
        wires = ins + list(s_reg[:nq]) + list(anc)
        qc.compose(sub, wires, inplace=True)
        qc.append(IntegerComparatorGate(nq, int(max(1 - inst.y0[t], 0)), geq=True), list(s_reg[:nq]) + [flags[t]])
        qc.compose(sub.inverse(), wires, inplace=True)
    qc.x(flags)
    qc.mcx(list(flags), final[0])
    qc.x(flags)
    qc.x(final[0])
    fidx = qc.find_bit(final[0]).index
    if flat:
        qc = transpile(qc, basis_gates=BASIS, optimization_level=1, qubits_initially_zero=True)
    return qc, fidx
