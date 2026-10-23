# Hot-swap socket for the SO-101 moving jaw

Same coordinate frame as the stock `Moving_Jaw_SO101.stl` (servo axis = Z, jaw points -Y,
grip face on the +X side). The fixed jaw on the wrist is untouched.

## socket.stl  (~18 g)
- Top is the stock jaw's ears + bridge, cut at y = -22.5. Bolts to the gripper servo horn
  exactly like the stock jaw (same four M3 holes per ear).
- Below the bridge: 16 x 22 mm block, 17.5 mm tall, with a blind pocket opening downward:
  8.3 x 10.3 mm, 15 mm deep (8 x 10 tang + 0.15 mm clearance per side).
- One pocket corner (+X, +Z) is chamfered 2 mm so an insert only goes in one way.
- Two 2.7 mm through-holes along Z at y = -29 and y = -35 for the M2.5 servo screws.
  Screw from either side; the screw self-taps into the tang.

## insert.stl  (~3 g)
- Tang 8 x 10 x 14.7 mm with the matching 2 mm key chamfer.
- Two 2.1 mm holes through the tang for the M2.5 screws to bite into.
- 16 x 22 x 4 mm flat plate on the end, flush with the socket block.
  **Model your attachment off the plate's bottom face (y = -44).**
  The stock grip face was on the +X side, tilted ~6 degrees toward -X going down the blade.

## Printing
- socket: stand it on the ear ends (horn end on the bed) with a brim, pocket opening up,
  PLA+, 0.2 mm, 40% infill. Tree supports if your slicer flags the ear tops.
- insert: plate flat on the bed, tang pointing up, 100% infill (it's tiny). No supports.

## Using the tang in CAD
Import insert.stl, or just rebuild it from the numbers above: it is a plain extruded
rectangle with one chamfered corner, so it is quicker to sketch than to import.

## launch_pad_insert.stl  (~13 g at 40% infill)
- insert.stl tang + plate, with a 90 x 90 x 2.5 mm pad hanging off the plate's bottom face
  (y = -44 to -46.5), 3 mm tall rim on the Tello side, four 26 mm weight-saving windows.
- Pad is centred 25 mm toward -X (away from the fixed jaw) so the open gripper clears it.
- Tello (98 x 92.5 mm with guards, 80 g) sits on the rimmed face; roll the wrist so the
  pad faces up before takeoff. Adjust PAD / X_OFF in make_launch_pad.py once the Tello is in
  hand if the guards foul the rim.
- Print pad face down on the bed, tang up, PLA+, 0.2 mm, 40% infill, no supports
  (windows and rim are on the bed side; tang needs none).
