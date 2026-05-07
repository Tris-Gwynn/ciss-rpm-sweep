# ciss_rpm/constants.py
import numpy as np
# Gyromagnetic ratio conversion factor (mT to frequency units)
# Assumed 28 MHz/mT based on your previous code
GYRO_E = 28 * 2 * np.pi

# Simulation settings
DEFAULT_STEPS = 250
RESOLUTION = 36
SWEEP_STEPS = 31
