# Speed-Mode Winder Control Reference Simulations

This folder contains reference simulations supporting the ORCA-VFD
research framework and the accompanying TechRxiv preprint:

"Field-Realistic Speed-Mode Winder Control: A Comparative Study"

The scripts compare three common winding control strategies using
speed-mode variable frequency drives:

1. Dancer-based speed trim control
2. Load-cell-based tension trim control
3. Sensorless tension estimation with speed trim control

These simulations are not part of the ORCA runtime or scoring pipeline.
They are provided as reproducible reference implementations illustrating
how feedback selection impacts regulation quality, actuator stress, and
estimation error under identical disturbance conditions.

All figures and metrics reported in the paper are generated directly
from these scripts.
