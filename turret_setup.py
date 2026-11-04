"""Turret servo setup: give the two new STS3215s their IDs (pan 7, tilt 8), centre them, test them.

Run from the Miniforge prompt (lerobot env) with the turret's Waveshare board on USB (jumper on B)
and the 12 V supply on. New servos all ship as ID 1, so plug in ONE servo at a time for `assign`.

    python turret_setup.py ports                      list COM ports (find the turret board)
    python turret_setup.py scan   --port COM9         show every servo on the bus
    python turret_setup.py assign --port COM9 --to 7  ONE servo plugged in: change its ID to 7
    python turret_setup.py center --port COM9 --id 7  move slowly to the middle (2048) and hold
    python turret_setup.py wiggle --port COM9 --id 7  small back-and-forth test, ends centred
    python turret_setup.py limits --port COM9         save the angle limits IN the servos (pan +-120, tilt +-40)
    python turret_setup.py teach  --port COM9         jog to your own left/right and up/down limits, save them in the servos
    python turret_setup.py test   --port COM9         slow pan + tilt sweep, Enter between steps, stops on a snag

Fit the horn (and later the turntable or the tilt hinge) while the servo is held at centre.
Ctrl+C at any time turns torque off.
"""
import argparse, json, os, sys, time
import serial  # pyserial, already in the lerobot env

BAUD = 1_000_000
ADDR_ID, ADDR_TORQUE, ADDR_ACC, ADDR_GOAL, ADDR_SPEED, ADDR_LOCK = 5, 40, 41, 42, 46, 55
ADDR_POS, ADDR_VOLT, ADDR_TEMP = 56, 62, 63
ADDR_MIN_LIM, ADDR_MAX_LIM, ADDR_LOAD = 9, 11, 60
CENTRE = 2048
STEPS_PER_DEG = 4096 / 360
PAN_ID, TILT_ID = 7, 8
PING, READ, WRITE = 0x01, 0x02, 0x03


class Bus:
    """Minimal Feetech STS protocol over the Waveshare board (FF FF id len instr params checksum)."""

    def __init__(self, port):
        try:
            self.s = serial.Serial(port, BAUD, timeout=0.03)
        except serial.SerialException as e:
            sys.exit(f"Can't open {port} ({e}). Check the port (python turret_setup.py ports) and close other programs using it.")

    def _tx(self, sid, instr, params=b""):
        body = bytes([sid, len(params) + 2, instr]) + bytes(params)
        pkt = b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])
        self.s.reset_input_buffer()
        self.s.write(pkt)
        return pkt

    def _rx(self, sid, n_params):
        want = 6 + n_params
        buf = b""
        t0 = time.time()
        while time.time() - t0 < 0.05:
            buf += self.s.read(want - len(buf) if len(buf) < want else 1)
            i = buf.find(b"\xff\xff" + bytes([sid]))
            if i >= 0 and len(buf) - i >= want:
                p = buf[i:i + want]
                if ((~sum(p[2:-1])) & 0xFF) == p[-1]:
                    return p[5:5 + n_params]
        return None

    def ping(self, sid):
        self._tx(sid, PING)
        return self._rx(sid, 0) is not None

    def read(self, sid, addr, size=1):
        self._tx(sid, READ, [addr, size])
        d = self._rx(sid, size)
        return None if d is None else int.from_bytes(d, "little")

    def write(self, sid, addr, val, size=1):
        self._tx(sid, WRITE, [addr] + list(int(val).to_bytes(size, "little")))
        if self._rx(sid, 0) is None:
            sys.exit(f"No reply from ID {sid} when writing register {addr}.")

    def close(self):
        self.s.close()


def scan(bus, quiet=False):
    found = [i for i in range(0, 21) if bus.ping(i)]
    if not quiet:
        if not found:
            print("No servos found. Check 12 V is on, the jumper is on B, and the servo cable is seated.")
        for i in found:
            v, t, p = bus.read(i, ADDR_VOLT), bus.read(i, ADDR_TEMP), bus.read(i, ADDR_POS, 2)
            print(f"  ID {i:2d}  position {p}  {v / 10 if v else '?'} V  {t} C")
    return found


def move(bus, sid, target, speed=600):
    bus.write(sid, ADDR_ACC, 30)
    bus.write(sid, ADDR_SPEED, speed, 2)
    bus.write(sid, ADDR_TORQUE, 1)
    bus.write(sid, ADDR_GOAL, target, 2)
    for _ in range(60):
        time.sleep(0.05)
        p = bus.read(sid, ADDR_POS, 2)
        if p is not None and abs(p - target) < 8:
            return p
    return bus.read(sid, ADDR_POS, 2)


def set_limits(bus, sid, deg):
    """Write min/max angle limits to the servo's EEPROM: it refuses any goal outside them, whatever program sends it."""
    lo, hi = CENTRE - round(deg * STEPS_PER_DEG), CENTRE + round(deg * STEPS_PER_DEG)
    bus.write(sid, ADDR_TORQUE, 0)
    bus.write(sid, ADDR_LOCK, 0)                 # unlock EEPROM
    bus.write(sid, ADDR_MIN_LIM, lo, 2)
    bus.write(sid, ADDR_MAX_LIM, hi, 2)
    time.sleep(0.05)
    bus.write(sid, ADDR_LOCK, 1)                 # lock (saved through power cycles)
    return bus.read(sid, ADDR_MIN_LIM, 2), bus.read(sid, ADDR_MAX_LIM, 2)


def move_watch(bus, sid, target, speed=150, max_load=450):
    """Slow move that stops where it is if the servo strains (a snagged cable, a part hitting something)."""
    bus.write(sid, ADDR_ACC, 10)
    bus.write(sid, ADDR_SPEED, speed, 2)
    bus.write(sid, ADDR_TORQUE, 1)
    start = bus.read(sid, ADDR_POS, 2) or target
    bus.write(sid, ADDR_GOAL, target, 2)
    t_end = time.time() + abs(target - start) / speed + 2.5
    while time.time() < t_end:
        time.sleep(0.05)
        p, ld = bus.read(sid, ADDR_POS, 2), bus.read(sid, ADDR_LOAD, 2)
        if ld is not None and (ld & 0x3FF) > max_load:      # 0.1 % units, bit 10 is direction
            if p is not None:
                bus.write(sid, ADDR_GOAL, p, 2)              # hold here, don't keep pushing
            return p, f"STOPPED: load {(ld & 0x3FF) / 10:.0f} %"
        if p is not None and abs(p - target) < 8:
            return p, "ok"
    return bus.read(sid, ADDR_POS, 2), "STOPPED: didn't reach the target"


LIMITS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "turret_limits.json")


def write_limits(bus, sid, lo, hi):
    bus.write(sid, ADDR_TORQUE, 0)
    bus.write(sid, ADDR_LOCK, 0)
    bus.write(sid, ADDR_MIN_LIM, lo, 2)
    bus.write(sid, ADDR_MAX_LIM, hi, 2)
    time.sleep(0.05)
    bus.write(sid, ADDR_LOCK, 1)
    return bus.read(sid, ADDR_MIN_LIM, 2), bus.read(sid, ADDR_MAX_LIM, 2)


def getkey():
    try:
        import msvcrt                      # Windows: single key presses
        return msvcrt.getwch()
    except ImportError:
        return (input() or "\r")[0]       # elsewhere: type the key + Enter


def jog_to(bus, sid, minus, plus, label):
    """Jog with the keys (lower case 2 deg, upper case 10 deg); Enter records the position, q aborts."""
    pos = bus.read(sid, ADDR_POS, 2) or CENTRE
    print(f"\n  Jog to the {label} limit: {minus}/{plus} = 2 deg, {minus.upper()}/{plus.upper()} = 10 deg, Enter = save, q = quit")
    while True:
        ch = getkey()
        if ch in ("\r", "\n"):
            p = bus.read(sid, ADDR_POS, 2)
            print(f"\n  {label} limit saved at {(p - CENTRE) / STEPS_PER_DEG:+.1f} deg")
            return p
        if ch == "q":
            return None
        step = {minus: -2, plus: 2, minus.upper(): -10, plus.upper(): 10}.get(ch)
        if step is None:
            continue
        pos = max(0, min(4095, pos + round(step * STEPS_PER_DEG)))
        p, status = move_watch(bus, sid, pos, speed=300)
        if p is not None:
            pos = p if status != "ok" else pos
            print(f"\r  {(p - CENTRE) / STEPS_PER_DEG:+6.1f} deg  {status}            ", end="", flush=True)


def teach(bus):
    saved = {}
    try:
        saved = json.load(open(LIMITS_FILE))
    except Exception:
        pass
    for sid, name, minus, plus, ends in ((PAN_ID, "PAN", "a", "d", ("LEFT", "RIGHT")),
                                          (TILT_ID, "TILT", "s", "w", ("DOWN", "UP"))):
        if not bus.ping(sid):
            print(f"ID {sid} ({name}) not found, skipped"); continue
        print(f"\n== {name} (ID {sid}) ==  watch the cables while you jog")
        write_limits(bus, sid, 0, 4095)              # open the range while teaching
        move_watch(bus, sid, CENTRE)
        a = jog_to(bus, sid, minus, plus, ends[0])
        if a is None: break
        move_watch(bus, sid, CENTRE)
        b = jog_to(bus, sid, minus, plus, ends[1])
        if b is None: break
        lo, hi = min(a, b), max(a, b)
        if not lo < CENTRE < hi:
            print(f"  Both limits are on the same side of home: home (centre) must be between them. {name} not saved.")
            write_limits(bus, sid, *saved.get(str(sid), (0, 4095)))
            continue
        lo, hi = write_limits(bus, sid, lo, hi)
        saved[str(sid)] = [lo, hi]
        print(f"  {name} limits saved in the servo: {(lo - CENTRE) / STEPS_PER_DEG:+.1f} to {(hi - CENTRE) / STEPS_PER_DEG:+.1f} deg")
        move_watch(bus, sid, CENTRE)
    json.dump(saved, open(LIMITS_FILE, "w"), indent=1)
    print(f"\nSaved to {os.path.basename(LIMITS_FILE)} too (turret_track keeps its aim inside them).")


def sweep(bus, sid, name, steps_deg):
    print(f"\n{name} (ID {sid}). Watch the cables. Enter = next step, q + Enter = stop.")
    for d in steps_deg:
        if input(f"  next: {name} to {d:+d} deg ").strip().lower() == "q":
            break
        p, status = move_watch(bus, sid, CENTRE + round(d * STEPS_PER_DEG))
        print(f"    at {(p - CENTRE) / STEPS_PER_DEG:+.1f} deg  {status}" if p is not None else f"    {status}")
        if status != "ok":
            print("    Check for a snagged cable or something touching, then rerun.")
            return False
    move_watch(bus, sid, CENTRE)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ports", "scan", "assign", "center", "wiggle", "limits", "teach", "test"])
    ap.add_argument("--port")
    ap.add_argument("--to", type=int, help="new ID for assign (7 = pan, 8 = tilt)")
    ap.add_argument("--id", type=int, help="servo ID for center / wiggle")
    ap.add_argument("--pan-max", type=int, default=120, help="pan limit, deg each way (cable loop: max 120)")
    ap.add_argument("--tilt-max", type=int, default=40, help="tilt limit, deg each way (40 laser plate, 20 Perses)")
    a = ap.parse_args()

    if a.cmd == "ports":
        from serial.tools import list_ports
        for p in list_ports.comports():
            print(f"  {p.device}  {p.description}")
        print("The arm is on COM8; the turret board is the other USB serial port (unplug it and rerun to see which disappears).")
        return
    if not a.port:
        sys.exit("Add --port COMx (python turret_setup.py ports lists them).")
    bus = Bus(a.port)
    sid = None
    try:
        if a.cmd == "scan":
            scan(bus)
        elif a.cmd == "assign":
            if a.to is None or not 1 <= a.to <= 20:
                sys.exit("Give --to 7 (pan) or --to 8 (tilt).")
            found = scan(bus, quiet=True)
            if len(found) != 1:
                sys.exit(f"Found servos {found or 'none'}. Plug in exactly ONE servo for assign.")
            old = found[0]
            if old == a.to:
                print(f"Already ID {a.to}."); return
            bus.write(old, ADDR_TORQUE, 0)
            bus.write(old, ADDR_LOCK, 0)          # unlock EEPROM
            bus.write(old, ADDR_ID, a.to)
            time.sleep(0.05)
            bus.write(a.to, ADDR_LOCK, 1)         # lock again (ID is now saved)
            ok = bus.ping(a.to)
            print(f"ID {old} -> {a.to}: {'done' if ok else 'FAILED, rerun scan'}")
            if ok:
                print("Label this servo, e.g. tape marked PAN 7 / TILT 8.")
        elif a.cmd == "limits":
            for sid_, deg in ((PAN_ID, min(a.pan_max, 120)), (TILT_ID, a.tilt_max)):
                if not bus.ping(sid_):
                    print(f"  ID {sid_} not found, skipped"); continue
                lo, hi = set_limits(bus, sid_, deg)
                print(f"  ID {sid_}: limits {lo}..{hi} (+-{deg} deg from centre), saved in the servo")
            print("The servos now refuse to move past these, even if a script asks them to.")
        elif a.cmd == "teach":
            sid = PAN_ID
            teach(bus)
        elif a.cmd == "test":
            ids = [i for i in (PAN_ID, TILT_ID) if bus.ping(i)]
            if not ids:
                sys.exit("No turret servos found (IDs 7, 8). Run scan.")
            for i in ids:
                lo, hi = bus.read(i, ADDR_MIN_LIM, 2), bus.read(i, ADDR_MAX_LIM, 2)
                if (lo, hi) == (0, 4095) or lo is None:
                    print(f"  ID {i} has no angle limits yet: run 'limits' first (recommended).")
            sid = ids[0]
            print("Both servos go to centre first, slowly.")
            for i in ids:
                move_watch(bus, i, CENTRE)
            def room(i, want):
                lo, hi = bus.read(i, ADDR_MIN_LIM, 2), bus.read(i, ADDR_MAX_LIM, 2)
                if lo is None or hi is None: return want, want
                return min(want, (CENTRE - lo) / STEPS_PER_DEG - 1), min(want, (hi - CENTRE) / STEPS_PER_DEG - 1)
            ok = True
            if TILT_ID in ids:
                dn, up = room(TILT_ID, a.tilt_max)
                ok = sweep(bus, TILT_ID, "TILT", [int(s_ * min(d, up if s_ > 0 else dn)) for d in (10, 20, 30, 40) for s_ in (1, -1)
                                                    if d <= max(up, dn) + 10])
            if ok and PAN_ID in ids:
                lf, rt = room(PAN_ID, min(a.pan_max, 120))
                sweep(bus, PAN_ID, "PAN", [int(s_ * min(d, rt if s_ > 0 else lf)) for d in (30, 60, 90, 120) for s_ in (1, -1)
                                            if d <= max(lf, rt) + 30])
            input("\nBack at centre. Press Enter to release (torque off: the tilt will sag a little)...")
        elif a.cmd in ("center", "wiggle"):
            if a.id is None:
                sys.exit("Give --id 7 or --id 8.")
            sid = a.id
            if not bus.ping(sid):
                sys.exit(f"No servo with ID {sid}. Run scan.")
            p = move(bus, sid, CENTRE)
            print(f"ID {sid} at {p} (centre {CENTRE}). Holding.")
            if a.cmd == "wiggle":
                for tgt in (CENTRE + 300, CENTRE - 300, CENTRE + 300, CENTRE):
                    p = move(bus, sid, tgt, speed=800)
                    print(f"  -> {tgt}: reached {p}")
                print("Wiggle OK, back at centre.")
            input("Fit the horn now if needed. Press Enter to release (torque off)...")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        for i in ({sid} if a.cmd not in ("test", "teach") else {PAN_ID, TILT_ID}):
            if i is None: continue
            try: bus._tx(i, WRITE, [ADDR_TORQUE, 0])
            except Exception: pass
        bus.close()


if __name__ == "__main__":
    main()
