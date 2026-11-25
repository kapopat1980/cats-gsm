import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import world, algos as A


def test_driving_trace_is_deterministic():
    a = world.driving_trace(1); b = world.driving_trace(1)
    assert np.allclose(a['x'], b['x']) and np.array_equal(a['serving'], b['serving'])


def test_driving_algorithms_run_and_energy_positive():
    tr = world.driving_trace(2, n=600)
    for r in (A.run_periodic(tr, 30), A.run_distance(tr, 250), A.run_dr_threshold(tr, 50),
              A.run_cats_drive(tr, 50), A.run_cellid(tr)):
        assert r['energy_J'] > 0 and r['mean'] >= 0


def test_replay_attack_is_caught_by_cats_not_by_gps_confirm():
    caught_cats = caught_a2 = 0
    for s in range(5):
        tr = world.armed_trace(3000 + s, 900, 1200, 'replay', acc_pd=1.0)
        caught_cats += any(a[0] >= tr['t0'] for a in A.armed_detect(tr, 'CATS')['alarms'])
        caught_a2 += any(a[0] >= tr['t0'] for a in A.armed_detect(tr, 'A2')['alarms'])
    assert caught_cats >= 4 and caught_a2 == 0


def test_no_theft_gives_few_alarms():
    tr = world.armed_trace(4000, 6 * 3600, 0, 'none', theft=False)
    assert len(A.armed_detect(tr, 'CATS', rearm=True)['alarms']) <= 3


def test_real_route_traces_run():
    import numpy as np
    n = 700; t = np.arange(n, dtype=float)
    x = 6000 + 10.0 * t; y = 6000 + 0 * t
    route = (x, y, np.full(n, 10.0), np.zeros(n), np.zeros(n), np.zeros(n))
    tr = world.driving_trace(5, route=route)
    assert tr['n'] == n and np.allclose(tr['x'], x)
    at = world.armed_trace(6, 300, 0, 'replay', route=route)
    assert at['n'] == 300 + n and abs(at['x'][0] - x[0]) < 1e-9
    assert A.run_cats_drive(tr, 100)['energy_J'] > 0
