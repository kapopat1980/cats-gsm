# CATS-GSM: cell-assisted asset tracking and vehicle security over GSM

Trace-driven Python simulator, baseline re-implementations and the **CATS-GSM** algorithm from the manuscript
*Cell-Assisted Security for GSM Vehicle Trackers: Detecting GPS Jamming and Spoofing with Serving-Cell and Timing-Advance Evidence*
(K. Popat, D. Meva, Marwadi University). Everything here is **simulation**. Study S uses synthetic grid-city drives; Study R replays real taxi routes (Porto, ECML/PKDD 2015).
The radio (serving cell, timing advance), IMU, GPS attacks and energy constants are modelled in both studies and are stated assumptions
(manuscript Sections 5 and 8). No field GSM or inertial measurements are used.

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
| `realdata.py` | Converts the Porto taxi trips (15 s GPS) into 1 Hz routes (Study R) |
| `exp_real.py` | Study R experiments on real routes (same schemes and statistics as `exp.py`) |
| `CATS_GSM_Porto_Kaggle.ipynb` | Self-contained Kaggle notebook that runs Study R end to end |
| `results_real/analysis_real.json` | Aggregated numbers behind Tables 8-9 |
| `results_real/results_real_raw.json.gz` | Per-trace raw results of Study R |
| `figures/fig_pareto_real.png` | Figure 2 |
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

### Study R (real routes)

Download the Porto taxi data (CC BY 4.0, 509 MB zip containing `train.csv.zip`) from
https://archive.ics.uci.edu/dataset/339/ , unzip `train.csv`, then:

```bash
PORTO_CSV=/path/to/train.csv python exp_real.py   # N_DRIVE=100 N_ARMED=150 N_FA=40 by default
python analysis.py                                  # writes analysis.json and fig_pareto.png
```

The notebook `CATS_GSM_Porto_Kaggle.ipynb` does the same on Kaggle. The trip data are not redistributed here.
Study R sampled trips at random (seed 0); results depend on the dataset file and sampling, so counts may differ slightly if you change `NROWS`.

All random draws are seeded (Study S: driving 1000-1099, theft 2000-2149, false alarms 9100-9139), so the numbers are
reproducible. Parameters tuned on data (the IMU speed threshold) were chosen on separate seeds 7000-7029 and 8000-8029.

### Robustness and tuned-baseline checks (v1.2.0)

Run from the repository root (`PYTHONPATH=. python diag/<script>.py`):

| Script | What it does |
| --- | --- |
| `diag/diag_resample.py` | Re-samples the synthetic drives to 15 s and re-interpolates them like the Porto routes; compares CATS-GSM, D3-IMU, D3-DR |
| `diag/tune_dr.py`, `diag/tune2.py` | Tunes the D3-DR constants (kappa, cap) on training seeds 7000-7029 |
| `diag/eval_tuned.py` | Evaluates tuned D3-DR on held-out synthetic and resampled routes |
| `diag/stress.py` | Armed-mode stress tests: shadowing, TA noise, hysteresis, thief speed |
| `exp_real_tuned.py` / `CATS_GSM_Porto_TunedDR_Kaggle.ipynb` | Re-runs the Porto driving comparison with the tuned D3-DR; output `results_real/tuned_driving.json` |

`world.py` now exposes `SIG_SP, SIG_TM, HYST, TA_NOISE, THIEF_F` (defaults reproduce all earlier results exactly).

## Cite

See `CITATION.cff`. Released under the MIT licence.
