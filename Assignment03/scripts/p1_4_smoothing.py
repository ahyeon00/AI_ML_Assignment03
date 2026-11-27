
import os
import sys
import numpy as np
from scipy.signal import savgol_filter

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, ROOT)

from utils.viser_test import PoseViser
from p1_2_post_processing import (
    create_height_corrected_target,
    export_unity_csv
)

# Data paths
DATA_DIR = os.path.join(ROOT, "data", "processed")

INPUT_PATH = os.path.join(
    DATA_DIR, "HAERIN_HowSweet_target.npy"
)

OUTPUT_PATH = os.path.join(
    DATA_DIR, "HAERIN_HowSweet_smoothed.npy"
)

CSV_PATH = os.path.join(
    DATA_DIR, "HAERIN_HowSweet_smoothed.csv"
)

# Load height-corrected motion
motion = np.load(INPUT_PATH)

# Savitzky-Golay smoothing
WINDOW_LENGTH = 7
POLYORDER = 2

smoothed = savgol_filter(
    motion,
    window_length=WINDOW_LENGTH,
    polyorder=POLYORDER,
    axis=0
)

# Restore ground alignment after smoothing
smoothed = create_height_corrected_target(smoothed)

# Save results
np.save(OUTPUT_PATH, smoothed)
export_unity_csv(smoothed, CSV_PATH)

print("\nSmoothing completed!")
print("Method: Savitzky-Golay Filter")
print("Window length:", WINDOW_LENGTH)
print("Polynomial order:", POLYORDER)
print("Data shape:", smoothed.shape)
print("Saved:", OUTPUT_PATH)

# Visualize smoothed motion
vis = PoseViser(fps=30)
vis.play_sequence(smoothed)
