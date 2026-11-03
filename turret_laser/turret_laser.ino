// Turret laser switch for an ESP32 DevKit. Laser + to VIN (5 V from USB, through the ARM toggle), laser - to a
// PN2222 collector, emitter to GND, base through 1 kOhm to GPIO 25. turret_track.py sends '1' (on) / '0' (off) at 10 Hz;
// if nothing arrives for 0.5 s (script crashed, cable out) the laser turns off by itself.
const int LASER = 25;   // GPIO 25 (was D2 on the Nano)
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
