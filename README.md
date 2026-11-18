# CATS-GSM: cell-assisted asset tracking and vehicle security over GSM

Trace-driven Python simulator, baseline re-implementations and the **CATS-GSM** algorithm from the manuscript
*Energy-Aware Adaptive Asset Tracking and Vehicle Security over GSM Cellular Networks*
(K. Popat, D. Meva, Marwadi University). Everything here is **simulation**: no field data are used, and the energy
constants and attack models are stated assumptions (manuscript Sections 5 and 7).

## Contents

| File | Purpose |
| --- | --- |
| `world.py` | City, base-station, GPS/IMU and attack models; trace generators (`driving_trace`, `armed_trace`) |
| `algos.py` | Baselines D1-D4 (driving), A1-A3 (armed), CATS-GSM, ablations, range-only EKF, energy meter |
| `exp.py` | Runs all experiments (100 driving traces, 4 attacks x 150 theft traces, 40 x 24 h false-alarm traces) |
| `analysis.py` | Bootstrap intervals, matched-power comparison, Wilcoxon/McNemar tests, Figure 1 |
| `sens_energy.py` | Sensitivity of the driving comparison to the GPRS and GPS energy constants (Table 7) |
| `results/analysis.json` | Aggregated numbers behind Tables 3-7 |
| `results/sens_energy.json` | Table 7 |
| `results/results_raw.json.gz` | Per-trace raw results from `exp.py` |
| `figures/fig_pareto.png` | Figure 1 |
| `tests/test_smoke.py` | Quick sanity tests |

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                 # a few seconds
python exp.py             # about 2 minutes on 2 cores; writes results.pkl
python analysis.py        # writes analysis.json and fig_pareto.png
python sens_energy.py     # writes sens_energy.json
```

All random draws are seeded (driving 1000-1099, theft 2000-2149, false alarms 9100-9139), so the numbers are
reproducible. Parameters tuned on data (the IMU speed threshold) were chosen on separate seeds 7000-7029 and 8000-8029.

## Cite

See `CITATION.cff`. Released under the MIT licence.
