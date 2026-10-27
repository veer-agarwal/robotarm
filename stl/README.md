# STL files to print

One folder per part set. Print settings in the README next to each design script.

| Folder | File | Print | Notes |
|---|---|---|---|
| gripper_socket | socket.stl | PLA+, 0.2 mm, 40% infill, brim, tree supports if flagged | Replaces Moving_Jaw_SO101.stl. Pocket opening up. ~18 g |
| gripper_socket | insert.stl | 100% infill, no supports, plate on bed | Blank tang + plate for custom attachments. ~3 g |
| launch_pad | launch_pad_insert.stl | 40% infill, no supports, pad face on bed, tang up | Tello launch pad on the socket tang. ~13 g |
| turret | base_floor, base_frame, cable_tray, turntable, ebox, ebox_tray, ebox_lid (v2), tower_servo, tower_bearing, laser_plate (now), cradle (Perses, later), servo_floor_cap_x2, stub_shaft, horn_spacer, roller_axle_x3, shaft_washer | See `turret/README.md` (orientation, infill, inserts, wiring) | Pan-tilt turret v2: flipped pan servo, electronics box on the turntable. Cradle is 250 mm long: X1E only |

Design scripts and previews live in `gripper/` (socket, insert, launch pad).
