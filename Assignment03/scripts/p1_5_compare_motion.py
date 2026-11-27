
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(ROOT, "data", "processed")
RESULT_DIR = os.path.join(ROOT, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

# Load motion data
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

titles = [
    "Original Motion",
    "Height Correction",
    "Correction + Smoothing"
]

colors = ["#E76F51", "#457BDE", "#2A9D8F"]

# BlazePose skeleton connections
connections = [
    (0, 7), (0, 8),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 29), (29, 31), (27, 31),
    (28, 30), (30, 32), (28, 32)
]

frames = min(len(m) for m in motions)
fps = 30

# Figure setup
fig = plt.figure(figsize=(15, 7))

fig.subplots_adjust(
    left=0.03,
    right=0.97,
    bottom=0.19,
    top=0.85,
    wspace=0.03
)

axes = [
    fig.add_subplot(1, 3, i + 1, projection="3d")
    for i in range(3)
]

fig.suptitle(
    "How Sweet - 3D Motion Post-Processing",
    fontsize=20,
    fontweight="bold"
)

frame_text = fig.text(
    0.5, 0.88, "",
    ha="center",
    fontsize=11
)

# Shared axis range
all_points = np.concatenate(motions, axis=0)

limit = max(
    0.8,
    np.max(np.abs(all_points[:, :, [0, 2]]))
)

y_min = min(-0.8, all_points[:, :, 1].min())
y_max = max(2.0, all_points[:, :, 1].max())

# Ground grid
gx, gz = np.meshgrid(
    np.linspace(-limit, limit, 11),
    np.linspace(-limit, limit, 11)
)

gy = np.zeros_like(gx)

# Draw motion
def draw_frame(frame):
    frame = int(frame)

    frame_text.set_text(
        f"Frame {frame + 1} / {frames}"
    )

    for ax, motion, title, color in zip(
        axes, motions, titles, colors
    ):
        ax.clear()

        pts = motion[frame]

        # Convert Y-up data to Matplotlib Z-up
        x = pts[:, 0]
        y = pts[:, 2]
        z = pts[:, 1]

        ax.plot_wireframe(
            gx, gz, gy,
            color="gray",
            alpha=0.2,
            linewidth=0.5
        )

        ax.scatter(
            x, y, z,
            color=color,
            s=14
        )

        for a, b in connections:
            ax.plot(
                [x[a], x[b]],
                [y[a], y[b]],
                [z[a], z[b]],
                color=color,
                linewidth=2.5
            )

        ax.set_title(
            title,
            fontsize=13,
            fontweight="bold",
            color=color
        )

        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(y_min, y_max)

        ax.set_box_aspect(
            (2 * limit, 2 * limit, y_max - y_min)
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Height (Y)")

        ax.view_init(elev=12, azim=-90)

    fig.canvas.draw_idle()

# Frame slider
slider_ax = fig.add_axes(
    [0.15, 0.10, 0.70, 0.03]
)

slider = Slider(
    slider_ax,
    "Frame",
    0,
    frames - 1,
    valinit=0,
    valstep=1
)

slider.on_changed(draw_frame)

# Playback state
playing = [True]

def animate(_):
    if playing[0]:
        slider.set_val(
            (int(slider.val) + 1) % frames
        )

# Play / Pause button
play_ax = fig.add_axes(
    [0.30, 0.025, 0.17, 0.045]
)

play_button = Button(
    play_ax, "Play / Pause"
)

def toggle_play(event):
    playing[0] = not playing[0]

play_button.on_clicked(toggle_play)

# Save comparison button
save_ax = fig.add_axes(
    [0.53, 0.025, 0.17, 0.045]
)

save_button = Button(
    save_ax, "Save Image"
)

def save_image(event):
    playing[0] = False

    path = os.path.join(
        RESULT_DIR,
        f"comparison_frame_{int(slider.val):04d}.png"
    )

    fig.savefig(path, dpi=200)
    print("Saved:", path)

save_button.on_clicked(save_image)

draw_frame(0)

animation = FuncAnimation(
    fig,
    animate,
    interval=1000 / fps,
    cache_frame_data=False
)

plt.show()
