"""Bench test for the blaster relays (ESP32 running turret_laser.ino). Keeps talking to the ESP32 10x a second,
because it switches everything off after 0.5 s of silence.

    python turret_blaster_test.py --port COM11 click     relays only (blaster unplugged): REV 1 s, then REV+FIRE 0.5 s
    python turret_blaster_test.py --port COM11 rev       flywheels on for 2 s
    python turret_blaster_test.py --port COM11 fire      rev 1 s, then fire for --fire seconds (default 0.3), then stop

Magazine OUT and safety glasses on for rev / fire. Ctrl+C stops everything (the ESP32 also stops by itself).
"""
import argparse, sys, time

try:
    import serial
except ImportError:
    sys.exit("pip install pyserial")


def run(s, cmd, secs):
    t = time.perf_counter()
    while time.perf_counter() - t < secs:
        s.write(cmd); time.sleep(0.1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["click", "rev", "fire"])
    ap.add_argument("--port", default="COM11")
    ap.add_argument("--fire", type=float, default=0.3, help="seconds of FIRE (about 7 darts/s)")
    a = ap.parse_args()
    s = serial.Serial(a.port, 115200, timeout=0)
    time.sleep(2.0)                                     # the ESP32 resets when the port opens
    try:
        if a.what == "click":
            print("REV on (relay 1 clicks)"); run(s, b"0Rf", 1.0)
            print("REV + FIRE (relay 2 clicks)"); run(s, b"0RF", 0.5)
        elif a.what == "rev":
            print("flywheels on 2 s"); run(s, b"0Rf", 2.0)
        else:
            print("rev 1 s"); run(s, b"0Rf", 1.0)
            print(f"FIRE {a.fire:.1f} s"); run(s, b"0RF", a.fire)
            run(s, b"0Rf", 0.3)
        print("off")
    except KeyboardInterrupt:
        print("stopped")
    finally:
        s.write(b"0rf"); time.sleep(0.1); s.close()


if __name__ == "__main__":
    main()
