// Turret laser switch for an Arduino Nano. Laser + to 5V (through the ARM toggle), laser - to a 2N2222
// collector, emitter to GND, base through 1 kOhm to D2. turret_track.py sends '1' (on) / '0' (off) at 10 Hz;
// if nothing arrives for 0.5 s (script crashed, cable out) the laser turns off by itself.
const int LASER = 2;
unsigned long last = 0;
void setup() { pinMode(LASER, OUTPUT); digitalWrite(LASER, LOW); Serial.begin(115200); }
void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '1') digitalWrite(LASER, HIGH);
    if (c == '0') digitalWrite(LASER, LOW);
    last = millis();
  }
  if (millis() - last > 500) digitalWrite(LASER, LOW);
}
