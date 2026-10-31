"""Turret servo setup: give the two new STS3215s their IDs (pan 7, tilt 8), centre them, test them.

Run from the Miniforge prompt (lerobot env) with the turret's Waveshare board on USB (jumper on B)
and the 12 V supply on. New servos all ship as ID 1, so plug in ONE servo at a time for `assign`.

    python turret_setup.py ports                      list COM ports (find the turret board)
    python turret_setup.py scan   --port COM9         show every servo on the bus
    python turret_setup.py assign --port COM9 --to 7  ONE servo plugged in: change its ID to 7
    python turret_setup.py center --port COM9 --id 7  move slowly to the middle (2048) and hold
    python turret_setup.py wiggle --port COM9 --id 7  small back-and-forth test, ends centred

Fit the horn (and later the turntable or the tilt hinge) while the servo is held at centre.
Ctrl+C at any time turns torque off.
"""
import argparse, sys, time
import serial  # pyserial, already in the lerobot env

BAUD = 1_000_000
ADDR_ID, ADDR_TORQUE, ADDR_ACC, ADDR_GOAL, ADDR_SPEED, ADDR_LOCK = 5, 40, 41, 42, 46, 55
ADDR_POS, ADDR_VOLT, ADDR_TEMP = 56, 62, 63
CENTRE = 2048
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ports", "scan", "assign", "center", "wiggle"])
    ap.add_argument("--port")
    ap.add_argument("--to", type=int, help="new ID for assign (7 = pan, 8 = tilt)")
    ap.add_argument("--id", type=int, help="servo ID for center / wiggle")
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
        if sid is not None:
            try: bus._tx(sid, WRITE, [ADDR_TORQUE, 0])
            except Exception: pass
        bus.close()


if __name__ == "__main__":
    main()
