
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(ROOT, "data", "processed")
RESULT_DIR = os.path.join(ROOT, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

# 1. Load motion data
raw = np.load(
    os.path.join(DATA_DIR, "HAERIN_HowSweet_raw.npy")
)

corrected = np.load(
    os.path.join(DATA_DIR, "HAERIN_HowSweet_target.npy")
)

smoothed = np.load(
    os.path.join(DATA_DIR, "HAERIN_HowSweet_smoothed.npy")
)

motions = [raw, corrected, smoothed]

names = [
    "Original",
    "Height Correction",
    "Correction + Smoothing"
]

colors = ["#E76F51", "#457BDE", "#2A9D8F"]

FPS = 30

# 2. Evaluation metrics
def foot_height(motion):
    return np.minimum(
        motion[:, 31, 1],
        motion[:, 32, 1]
    )

def ground_error(motion):
    return np.mean(
        np.abs(foot_height(motion))
    )

def frame_displacement(motion):
    diff = np.diff(motion, axis=0)
    return np.linalg.norm(diff, axis=2).mean(axis=1)

def motion_roughness(motion):
    diff2 = np.diff(motion, n=2, axis=0)
    return np.linalg.norm(diff2, axis=2).mean()

def roughness_sequence(motion):
    diff2 = np.diff(motion, n=2, axis=0)
    return np.linalg.norm(diff2, axis=2).mean(axis=1)

def improvement(before, after):
    if before == 0:
        return 0.0
    return (before - after) / before * 100


# 3. Calculate results
results = []

for name, motion in zip(names, motions):
    results.append({
        "Method": name,
        "Frames": len(motion),
        "Ground Error (m)": ground_error(motion),
        "Mean Displacement (m)": frame_displacement(motion).mean(),
        "Motion Roughness (m)": motion_roughness(motion)
    })

df = pd.DataFrame(results)

print("\n===== MOTION ANALYSIS =====")
print(df.to_string(index=False))

ground_improvement = improvement(
    ground_error(raw),
    ground_error(corrected)
)

smoothing_improvement = improvement(
    motion_roughness(corrected),
    motion_roughness(smoothed)
)

print("\n===== IMPROVEMENT =====")

print(
    f"Ground alignment improvement: "
    f"{ground_improvement:.2f}%"
)

print(
    f"Roughness reduction after smoothing: "
    f"{smoothing_improvement:.2f}%"
)

df.to_csv(
    os.path.join(RESULT_DIR, "analysis_results.csv"),
    index=False
)

# 4. Graph styling
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False
})

def save_graph(filename):
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULT_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )

# 5. Foot height comparison
plt.figure(figsize=(11, 4.5))

for motion, name, color in zip(motions, names, colors):
    plt.plot(
        foot_height(motion),
        label=name,
        color=color,
        linewidth=1.7
    )

plt.axhline(
    0, color="black",
    linestyle="--",
    linewidth=1
)

plt.title(
    "Foot Height Before and After Correction",
    fontweight="bold"
)

plt.xlabel("Detected Frame")
plt.ylabel("Foot Height (m)")
plt.legend()
plt.grid(alpha=0.15)

save_graph("01_foot_height.png")

# 6. Right wrist trajectory
plt.figure(figsize=(11, 4.5))

start = 0
end = min(150, len(raw))

for motion, name, color in zip(
    motions[1:], names[1:], colors[1:]
):
    plt.plot(
        np.arange(start, end),
        motion[start:end, 16, 0],
        label=name,
        color=color,
        linewidth=1.6
    )

plt.title(
    "Right Wrist X-Axis Trajectory",
    fontweight="bold"
)

plt.xlabel("Detected Frame")
plt.ylabel("X Position (m)")
plt.legend()
plt.grid(alpha=0.15)

save_graph("02_wrist_trajectory.png")

# 7. Roughness comparison
plt.figure(figsize=(8, 5))

values = [
    motion_roughness(m) for m in motions
]

bars = plt.bar(
    ["Original", "Height\nCorrection",
     "Correction\n+ Smoothing"],
    values,
    color=colors,
    width=0.55
)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.5f}",
        ha="center",
        va="bottom"
    )

plt.ylim(0, max(values) * 1.25 + 1e-8)

plt.title(
    "Motion Roughness Comparison",
    fontweight="bold"
)

plt.ylabel("Mean Second Difference (m)")
plt.grid(axis="y", alpha=0.15)

save_graph("03_roughness_comparison.png")

# 8. Temporal roughness
plt.figure(figsize=(11, 4.5))

for motion, name, color in zip(
    motions[1:], names[1:], colors[1:]
):
    plt.plot(
        roughness_sequence(motion),
        label=name,
        color=color,
        linewidth=1.2,
        alpha=0.85
    )

plt.title(
    "Temporal Motion Roughness",
    fontweight="bold"
)

plt.xlabel("Detected Frame")
plt.ylabel("Second Difference (m)")
plt.legend()
plt.grid(alpha=0.15)

save_graph("04_temporal_roughness.png")

print("\nAll graphs saved successfully.")
print("Results folder:", RESULT_DIR)

plt.show()
