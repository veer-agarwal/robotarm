# Turret simulator

Tests aiming code against a simulated Tello before the real camera is on the turret.

    python turret_sim/run_sim.py                     every flight path x every controller (servo ACC 254)
    python turret_sim/run_sim.py --acc 527           the same with today's ACC 60
    python turret_sim/run_sim.py --video --only zigzag --ctrl current kalman_auto

What's modelled:
- **Camera:** Arducam OV9281 with the 130° lens (640x400 @ 120 fps, 2 ms exposure, 25 ms latency). Mono, with noise and motion blur.
- **Servos:** STS3215 at SPEED 2500 (220°/s), with ACC as set, gear backlash and limits.
- **Room:** a grey panorama with posters, lights and a window.
- **Flight paths:** hover, crossing, zigzag, jink, orbit and close_pass (1.2 m away at 4 m/s).
- **Timing:** each controller gets the newest frame once it's free, and its real compute time counts.

The predictive controller is `turret_predict.py` in the repo root, the same file `turret_track.py --mode predict`
runs, so a fix made here is a fix on the turret.

## Results (on target = aim inside the drone's outline; first 0.5 s skipped)

Servo ACC 254 (2230°/s²):

| path       | today's follow | predict (auto) |
|------------|---------------:|---------------:|
| hover      | 100%           | 100%           |
| crossing   | 50%            | 100%           |
| zigzag     | 27%            | 98%            |
| jink       | 83%            | 98%            |
| orbit      | 32%            | 100%           |
| close_pass | 24%            | 98%            |

At ACC 60 the same zigzag drops to 1% (follow) and 16% (predict). The servos can't change direction fast enough,
so raising ACC matters as much as the software.

Lead 30 ms and q 2e6 came from a grid search. Leads of 45-60 ms overshoot.

## Known limits
- The simulated drone is dark against the background, which suits the blob tracker. A real Tello may not be.
- The template tracker (the fallback for targets that aren't dark) loses the target on a close pass, because the
  drone grows about 3x. Worse, it can look locked while it's wrong. Next step: a trained drone detector, using
  clips recorded with R in turret_track.py.
- `--latency` must match the real camera. In tests, a 15 ms error made the turret wobble. Measure it once the
  Arducam is here.
