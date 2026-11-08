"""The Island Interconnected System supply fleet and the portfolios we test.

Capacities (firm MW) and forced-outage assumptions are NL Hydro's own planning values, filed with the
Public Utilities Board (PUB):
  [RAP24] 2024 Resource Adequacy Plan, Rev. 2 (2024-08-26)            capacities, firm values, LOLH criterion
  [NTR25] 2025 Near-Term Reliability Report (2025-11-20)              outage rates, LIL 700 MW / 5% EqFOR, LOLH
  [Q3-26] Q3 2026 Rolling 12-Month Asset Performance (2026-11-02)     actual outage rates
  [WR26]  2026-27 Winter Readiness Report, October update (2026-10-13)
  [LF25]  2025 Island Load Forecast (2025-11-05)                       peak forecasts
  [ACT26] Avalon Combustion Turbine application, Rev. 1 (2026-04-16)  replacement plan
  [ELCC25] Effective Load Carrying Capability Study (2025-12-09)      wind / battery capacity values
URLs are listed in the notebook and README.

Simplifications (stated in the notebook): planned maintenance is ignored (winter peak focus), hydro energy
limits are ignored, the Labrador-Island Link (LIL) is one 700 MW unit whose 5% equivalent forced outage
rate is treated as full-bipole outage probability, and imports over the Maritime Link count as zero (Hydro
counts no firm imports [NTR25]).
"""
from __future__ import annotations

from .reliability import Portfolio, Unit

HYDRO_DAFOR = 0.025          # regulated hydro, near-term [NTR25]
THIRD_PARTY_HYDRO = 0.049    # [NTR25]
HOLYROOD_DAUFOP = 0.20       # planning value; 34% sensitivity; actual 27% in the 12 months to Sep 2026 [Q3-26]
CT_DAUFOP = 0.049            # Holyrood CT [NTR25]
OLD_GT_DAUFOP = 0.30         # Hardwoods, Stephenville, NP Greenhill / Wesleyville [NTR25]
DIESEL = 0.064               # [NTR25]
CBPP = 0.192                 # [NTR25]
LIL_MW, LIL_EQFOR = 700.0, 0.05   # planning model [NTR25]; rated 900 MW

LOLH_CRITERION_H = 2.8       # hours/year [RAP24]
PEAK_FORECAST_MW = {2026: 1788, 2027: 1803, 2028: 1818, 2029: 1841, 2030: 1860}   # incl. losses [NTR25]
HYDRO_LOLH_2026 = 1.8        # NL Hydro's own result for 2026, Reference case [NTR25]
HOLYROOD_CO2_KT = {2022: 655, 2023: 608, 2024: 593}   # Holyrood TGS, CO2e [ECCC/NL facility data]


def _hydro():
    u = [Unit(f"Bay d'Espoir U{i}", 76.5, HYDRO_DAFOR, "hydro", cold_sensitive=True) for i in range(1, 7)]
    u += [Unit("Bay d'Espoir U7", 154.4, HYDRO_DAFOR, "hydro", cold_sensitive=True),
          Unit("Upper Salmon", 84, HYDRO_DAFOR, "hydro"), Unit("Granite Canal", 40, HYDRO_DAFOR, "hydro"),
          Unit("Hinds Lake", 75, HYDRO_DAFOR, "hydro"), Unit("Cat Arm U1", 67, HYDRO_DAFOR, "hydro"),
          Unit("Cat Arm U2", 67, HYDRO_DAFOR, "hydro"), Unit("Paradise River", 8, HYDRO_DAFOR, "hydro"),
          Unit("Exploits (firm)", 63, THIRD_PARTY_HYDRO, "hydro", cold_sensitive=True),
          Unit("Star Lake", 18, THIRD_PARTY_HYDRO, "hydro"), Unit("Deer Lake Power", 104, THIRD_PARTY_HYDRO, "hydro")]
    u += [Unit(f"NP hydro {i}", 60.1 / 6, THIRD_PARTY_HYDRO, "hydro") for i in range(6)]
    return u


def _thermal(holyrood: bool = True, stephenville: bool = True, np_thermal_mw: float = 44.5):
    u = []
    if holyrood:
        u += [Unit("Holyrood U1", 170, HOLYROOD_DAUFOP, "thermal_oil", "No. 6 fuel oil"),
              Unit("Holyrood U2", 170, HOLYROOD_DAUFOP, "thermal_oil", "No. 6 fuel oil"),
              Unit("Holyrood U3", 150, HOLYROOD_DAUFOP, "thermal_oil", "No. 6 fuel oil")]
    u += [Unit("Holyrood CT", 123.5, CT_DAUFOP, "ct", "diesel"), Unit("Hardwoods GT", 50, OLD_GT_DAUFOP, "ct", "diesel")]
    if stephenville:
        u += [Unit("Stephenville GT", 50, OLD_GT_DAUFOP, "ct", "diesel")]
    u += [Unit("NP thermal", np_thermal_mw, OLD_GT_DAUFOP, "ct", "diesel"),
          Unit("Hydro diesels", 22.7, DIESEL, "diesel", "diesel"),
          Unit("CBPP co-gen", 8, CBPP, "cogen", "biomass"),
          Unit("CBPP capacity assistance", 90, 0.05, "demand_response")]
    return u


def lil(mw: float = LIL_MW, eqfor: float = LIL_EQFOR) -> Unit:
    return Unit("Labrador-Island Link", mw, eqfor, "hvdc", cold_sensitive=True)


def today() -> Portfolio:
    """Winter 2026-27 system: Holyrood available, existing 54 MW of wind (output from weather)."""
    return Portfolio("2026-27 system (with Holyrood oil)", _hydro() + _thermal() + [lil()], wind_mw=54.0)


def no_holyrood_2030() -> Portfolio:
    """2030: Holyrood steam units and Stephenville GT retired, NP thermal reduced [RAP24]; wind PPAs renewed."""
    return Portfolio("2030, Holyrood oil retired, nothing added",
                     _hydro() + _thermal(holyrood=False, stephenville=False, np_thermal_mw=16.5) + [lil()], wind_mw=54.0)


def hydro_plan_2030() -> Portfolio:
    """Hydro's plan: new 150 MW Avalon CT (diesel, LM6000s at Holyrood) and Bay d'Espoir Unit 8 (150 MW) [ACT26]."""
    p = no_holyrood_2030()
    return p.plus(Unit("Avalon CT (new, diesel)", 150, CT_DAUFOP, "ct", "diesel"),
                  Unit("Bay d'Espoir U8 (new)", 150, HYDRO_DAFOR, "hydro", cold_sensitive=True),
                  name="2030, Hydro's plan: new Avalon diesel CT + Bay d'Espoir U8")


def battery(mw: float) -> Unit:
    return Unit(f"Battery {mw:.0f} MW / 4 h", mw, 0.03, "battery")
