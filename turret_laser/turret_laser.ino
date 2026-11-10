// Turret laser + blaster switch for an ESP32 DevKit.
//
// Laser:  laser + to VIN (5 V from USB, through the ARM toggle), laser - to a PN2222 collector, emitter to GND,
//         base through 1 kOhm to GPIO 25.
// Blaster: 2-channel relay module. IN1 = GPIO 26 (REV, flywheels), IN2 = GPIO 27 (FIRE, pusher). Each relay's
//         COM + NO go across one of the blaster's microswitches (REV relay across the flywheel switch, FIRE relay
//         across the pusher/trigger switch). Relay module VCC = 5 V (VIN), GND = GND. If your module has a
//         high/low trigger jumper, set it to HIGH and leave RELAY_ACTIVE_LOW false.
//
// turret_track.py sends characters at 10 Hz:  '1'/'0' laser on/off,  'R'/'r' rev on/off,  'F'/'f' fire on/off.
// Interlocks that hold even if the PC software misbehaves:
//   - nothing received for 0.5 s (script crashed, cable out): everything off
//   - FIRE only works once REV has been on for SPINUP_MS (no pushing darts into stopped flywheels)
//   - FIRE on for more than MAX_FIRE_MS continuously: FIRE drops and stays off until an 'f' arrives
const int LASER = 25, REV = 26, FIRE = 27;
const bool RELAY_ACTIVE_LOW = false;
const unsigned long SILENCE_MS = 500, SPINUP_MS = 250, MAX_FIRE_MS = 2000;

bool laserOn = false, revOn = false, fireReq = false, fireOn = false, fireLocked = false;
unsigned long last = 0, revSince = 0, fireSince = 0;

void relay(int pin, bool on) { digitalWrite(pin, (on != RELAY_ACTIVE_LOW) ? HIGH : LOW); }

void allOff() { laserOn = revOn = fireReq = fireOn = false; }

void setup() {
  pinMode(LASER, OUTPUT); pinMode(REV, OUTPUT); pinMode(FIRE, OUTPUT);
  digitalWrite(LASER, LOW); relay(REV, false); relay(FIRE, false);
  Serial.begin(115200);
}

void loop() {
  unsigned long now = millis();
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '1') laserOn = true;
    if (c == '0') laserOn = false;
    if (c == 'R' && !revOn) { revOn = true; revSince = now; }
    if (c == 'r') revOn = false;
    if (c == 'F') fireReq = true;
    if (c == 'f') { fireReq = false; fireLocked = false; }
    last = now;
  }
  if (now - last > SILENCE_MS) allOff();

  bool want = fireReq && revOn && !fireLocked && (now - revSince >= SPINUP_MS);
  if (want && !fireOn) fireSince = now;
  if (want && now - fireSince > MAX_FIRE_MS) { want = false; fireLocked = true; }
  fireOn = want;

  digitalWrite(LASER, laserOn ? HIGH : LOW);
  relay(REV, revOn);
  relay(FIRE, fireOn);
}
