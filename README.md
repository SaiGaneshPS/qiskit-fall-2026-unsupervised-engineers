# Lights On, Oil Off

**Does Newfoundland need a new oil-burning turbine to retire Holyrood?**
Qiskit Fall Fest 2026 (Memorial University): *Quantum for a Sustainable Newfoundland and Labrador*.

Holyrood burns heavy oil (about 600 kt CO2 a year) and is due to retire around 2030. NL Hydro's replacement plan is
Bay d'Espoir Unit 8 plus a 150 MW diesel combustion turbine at Holyrood. We built an island grid model from public data
(an AI demand model, 46 replayed winters, a reliability engine checked against NL Hydro's own numbers) and asked whether
wind and heat pumps can do the turbine's job while keeping expected blackouts under 2.8 hours a year. Every answer is
a small probability near that limit, so we also built the risk as a quantum circuit and estimated it with quantum
amplitude estimation, in simulation and on IBM's ibm_quebec chip.

Start with `notebook.ipynb`.

## Results (simulators unless stated)

| Question | Result |
|---|---|
| Do we need the turbine? | At the Labrador-Island Link's measured outage rate (1.1%), no: Unit 8 + 200 MW wind + 90,000 heat pumps + smart thermostats passes (1.75 h/yr) at about the same yearly cost as the cheapest passing plan with the turbine (CAD 66.6M vs 64.4M) |
| What does the turbine buy? | Tolerance for a bad link: our strongest plan without it is safe up to a 3.3% link outage rate; with the turbine, beyond 5% (Hydro's planning value). Hydro's two projects alone: 1.6% |
| Can we trust the model? | 2026-27 blackout hours: ours 1.63 h vs NL Hydro 1.8 h; with Holyrood at 34% outage rate, 3.74 h vs 4.1 h |
| AI demand model, held-out 2025-26 winter | MAPE 5.7% vs 6.6% for the utility-standard degree-hour model; on hours colder than any training data, 41 vs 80 MW error |
| Heat pumps | 40k oil homes: about -260 kt CO2/yr but blackout hours 1.6 -> 2.7; 50k baseboard homes: blackout hours halved (1.6 -> 0.8) |
| Quantum circuit correct? | 24-qubit circuit = exact enumeration (5.86e-4 per winter hour); amplitude estimation 4.4% error at 13,300 samples vs 35.8% for Monte Carlo |
| Samples for a confident pass/fail call near the limit | amplitude estimation (ideal) 24k vs 82k tuned importance sampling vs 2.1M Monte Carlo (link 3.4%); far from the limit importance sampling is cheaper |
| Samples at 10% precision | 4x to 48x fewer than Monte Carlo, growing like p^-0.47 (theory p^-0.5); about par with tuned importance sampling |
| Real hardware (ibm_quebec) | the boosted circuits work; with twin calibration, a tie with Monte Carlo on shallow circuits (9.1% vs 10.7% error at 4,000 runs); Monte Carlo wins on deeper circuits or if calibration runs are counted |
| The bottleneck: plans through a cold snap | 4 plans x 2 link rates on the real 22-qubit circuits (Aer): quantum lands on the exact answer every time, 2 to 3.5x more precise than Monte Carlo with the same runs; on the hardest call (9.5% vs a 10% line) right in 184 of 200 tries vs 34 for Monte Carlo |
| 3-day cold snap | 22-qubit chronological circuit matches exact enumeration (0.0961); treating days as independent gives 0.147 |
| What didn't work | quantum-enhanced search over the 4,096 plans (slower in 41 of 45 tests); deep circuits on hardware; no model predicts the Feb 2023 flash-freeze record |

## Repository

| Path | Contents |
|---|---|
| `notebook.ipynb` | Working notebook with saved outputs (start here) |
| `demo/index.html` | Working demo: build a 2030 plan, set the link outage rate, see blackout hours and cost, and compare Monte Carlo with quantum amplitude estimation live. Open it in any browser; data from `tools/export_demo_data.py` |
| `gridrisk/` | Data download, AI demand model, fleet, reliability engine, heat-pump scenarios, 2030 plans, quantum circuits, plots, notebook tables |
| `experiments/` | E1 AI demand model, E2 reliability and heat pumps, E3 24-qubit circuit on Aer, E4 samples vs rarity, E5 hardware (noise model and ibm_quebec), E6 3-day cold snap, E7 all 4,096 plans, `turbine_decision.py` link break-even and samples per decision. The `e8*` files are a side study on rotating outages that is not part of this submission |
| `data/raw/` | Cached public inputs (re-downloaded automatically if missing) |
| `results/`, `figures/` | Saved outputs and figures |
| `tools/build_notebook.py` | Generates the notebook |
| `slides/` | Presentation: `Lights_On_Oil_Off.pptx` and a PDF copy. `source/` holds an earlier draft |

## Run

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebook.ipynb   # set RECOMPUTE = True to rerun every experiment (about 45 minutes)
```

Real hardware (optional; needs an IBM Quantum account saved with `QiskitRuntimeService.save_account`):
`python experiments/e5_hardware_scale.py --hardware --backend ibm_quebec` (add `--twin` for the calibrated run)

## Credits

NL Hydro data via Statistics Canada / CCEI High-Frequency Electricity Data; ERA5 via Open-Meteo; NL Hydro filings to the
Public Utilities Board; Statistics Canada; NRCan, NRC and NEEP heat pump data; ECCC emission factors. Amplitude
estimation (Brassard et al. 2002), maximum likelihood (Suzuki et al. 2020), noise-aware (Tanaka et al. 2021); quantum
risk analysis (Woerner and Egger 2019; Egger et al. 2021). AI tools assisted with code and documentation; the team
reviewed and can explain all of it.
