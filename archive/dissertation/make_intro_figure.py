"""Chapter 1 illustrative figure: the two communication interfaces, conceptually.

Deliberately schematic, not data-driven -- this figure supports the Chapter 1
illustrative example, not the real Chapter 3/4 results. Colour scheme matches
fig3_1_pipeline.png (Arm A blue, Arm B orange) for visual consistency across
the dissertation's two pipeline diagrams.

Horizontal (left-to-right) layout: ECG -> CNN-LSTM -> split into an Arm A row
(top) and an Arm B row (bottom) -> both converge into Gemma -> response.
Arm B has one more processing step than Arm A, so its row extends one stage
further right before the two converge -- mirrors the original vertical
version's column-length asymmetry, just rotated.

Run (from repo root):
    python make_intro_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

OUT_DIRS = [r"D:\Dissertation\latex_src\figures", r"D:\Dissertation\latex_build\figures"]

BLUE_DARK = "#1868A3"
BLUE_LIGHT = "#4A90C9"
ORANGE_DARK = "#D2581A"
ORANGE_LIGHT = "#F0AC82"
GEMMA_GREY = "#3D3D3D"
TEXT_GREY = "#333333"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 12.5})


def box(ax, xy, w, h, text, fc, tc="white", fs=13, weight="bold"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                 boxstyle="round,pad=0.02,rounding_size=0.08",
                                 fc=fc, ec="none", zorder=2))
    ax.text(x, y, text, ha="center", va="center", color=tc, fontsize=fs,
             fontweight=weight, zorder=3, linespacing=1.35)


def arrow(ax, p0, p1, color=TEXT_GREY, lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, color=color,
                                  lw=lw, mutation_scale=14, zorder=1,
                                  shrinkA=2, shrinkB=2))


REAL_ECG_IDX = 1905  # DS2 index, real ventricular-ectopic (V-class) beat


def real_ecg_window(idx=REAL_ECG_IDX, ds2_path="data/processed/ds2_test.npz"):
    """One real, R-peak-centred 360-sample ECG window from the actual MIT-BIH
    DS2 test set. idx=1905 is a true V-class beat the trained Perception Agent
    also classifies as V (matches the figure's narrative direction), at a real
    confidence of 0.533 -- deliberately far from the illustrative 0.91 shown
    in the boxes, so the real waveform and the illustrative values cannot be
    mistaken for one another. Not reused anywhere else in this dissertation."""
    w = np.load(ds2_path)["features"][idx].astype(float)
    w = (w - w.min()) / (w.max() - w.min())
    return np.arange(len(w)), w


fig, ax = plt.subplots(figsize=(17, 8))
ax.set_xlim(0, 19.5)
ax.set_ylim(0, 10)
ax.axis("off")

Y_TOP, Y_MID, Y_BOT = 7.3, 5.0, 2.7

# --- ECG trace panel (feeds the shared spine) -------------------------------
ecg_x0, ecg_y0, ecg_w, ecg_h = 1.9, Y_MID, 3.1, 2.0
ax.add_patch(FancyBboxPatch((ecg_x0 - ecg_w/2, ecg_y0 - ecg_h/2), ecg_w, ecg_h,
                             boxstyle="round,pad=0.02,rounding_size=0.06",
                             fc="#FBFBFB", ec="#CCCCCC", lw=1.0, zorder=2))
t, y = real_ecg_window()
tx = ecg_x0 - ecg_w/2 + 0.2 + (t / t.max()) * (ecg_w - 0.4)
ty = (ecg_y0 - ecg_h/2 + 0.35) + y * (ecg_h - 1.0)
ax.plot(tx, ty, color="#1a1a1a", lw=1.1, zorder=3)
ax.text(ecg_x0, ecg_y0 + ecg_h/2 - 0.16, "MIT-BIH ECG segment",
        ha="center", va="top", fontsize=10, color=TEXT_GREY, style="italic")
ax.text(ecg_x0, ecg_y0 + ecg_h/2 - 0.42, "(DS2, ventricular beat)",
        ha="center", va="top", fontsize=10, color=TEXT_GREY, style="italic")

# --- Shared spine: ECG -> CNN-LSTM -> split ---------------------------------
arrow(ax, (ecg_x0 + ecg_w/2, Y_MID), (3.9, Y_MID))
box(ax, (4.9, Y_MID), 2.1, 1.5, "CNN-LSTM\nECG model", GEMMA_GREY, fs=13)

arrow(ax, (5.95, Y_MID + 0.35), (6.7, Y_TOP - 0.05))
arrow(ax, (5.95, Y_MID - 0.35), (6.7, Y_BOT + 0.05))

# --- Row headers + one-line "what's different, at a glance" summaries ------
# Positioned relative to each row's TALLEST box edge (Arm A's 4-line box is
# 1.85 tall, Arm B's first box only 1.0) -- using one fixed offset for both,
# as an earlier version did, put Arm A's text inside its own box.
arm_a_box_top = Y_TOP + 1.85 / 2
arm_b_box_bottom = Y_BOT - 1.0 / 2
ax.text(7.5, arm_a_box_top + 0.5, "ARM A", ha="center", fontsize=13, fontweight="bold", color=BLUE_DARK)
ax.text(7.5, arm_a_box_top + 0.2, "Prediction + metadata $\\rightarrow$ JSON text",
        ha="center", fontsize=9.5, color=BLUE_DARK, style="italic")
ax.text(7.5, arm_b_box_bottom - 0.2, "32-D representation $\\rightarrow$ 4 virtual tokens",
        ha="center", fontsize=9.5, color=ORANGE_DARK, style="italic")
ax.text(7.5, arm_b_box_bottom - 0.5, "ARM B", ha="center", fontsize=13, fontweight="bold", color=ORANGE_DARK)

# Both arms start at the same x (7.5, right after the split) and END at the
# same x (14.6, right before converging into Gemma) regardless of how many
# boxes each one has in between. Arm A genuinely has fewer real steps than
# Arm B (predict -> write JSON, vs. predict -> project -> 4 tokens), so it
# gets one long arrow instead of a padded extra box -- an earlier version
# instead let Arm A's chain end early, leaving a long empty stretch on the
# top row while Arm B's extra box filled the same span, which read as
# unbalanced. Matching start/end x fixes that without inventing a fake step.
ARM_END_X = 14.6

# --- Arm A row (top) ---------------------------------------------------------
# Fields match reasoning/prompt_template.py's _extract_prompt_fields() subset
# actually sent to Gemma -- NOT requires_urgent_review/flag_reason, which are
# deliberately withheld as the scoring ground truth. "Signal features" is the
# real transmitted field group; "clinical flags" would misstate the interface.
box(ax, (7.5, Y_TOP), 3.4, 1.85,
    "Class: V\nConfidence: 0.91\nTop-3: V, N, F\nSignal features: …",
    BLUE_LIGHT, tc="#0A2E4A", fs=11, weight="normal")
arrow(ax, (9.2, Y_TOP), (ARM_END_X - 1.25, Y_TOP))
box(ax, (ARM_END_X, Y_TOP), 2.5, 0.95, "JSON message", BLUE_DARK, fs=12)

# --- Arm B row (bottom) ------------------------------------------------------
box(ax, (7.5, Y_BOT), 3.0, 1.0, "32-dimensional\nrepresentation", ORANGE_LIGHT,
    tc="#5C2A00", fs=11.5)
arrow(ax, (9.0, Y_BOT), (10.05, Y_BOT))
box(ax, (11.05, Y_BOT), 2.6, 0.85, "Learned adapter", ORANGE_DARK, fs=12)
arrow(ax, (12.35, Y_BOT), (ARM_END_X - 1.3, Y_BOT))
box(ax, (ARM_END_X, Y_BOT), 2.6, 0.95, "4 virtual tokens", ORANGE_DARK, fs=12)

# --- Converge into Gemma -- both arms now start their final arrow from the
# same x, so the two convergence arrows are symmetric in length and angle ---
arrow(ax, (ARM_END_X + 1.25, Y_TOP), (17.05, Y_MID + 0.45))
arrow(ax, (ARM_END_X + 1.3, Y_BOT), (17.05, Y_MID - 0.45))

box(ax, (18.2, Y_MID), 1.9, 1.5, "Gemma", GEMMA_GREY, fs=15)
arrow(ax, (19.15, Y_MID), (19.85, Y_MID))

# Response box sits just outside the main axes width budget on purpose --
# extend xlim slightly instead of shrinking everything else to fit it in.
ax.set_xlim(0, 22.5)
box(ax, (20.9, Y_MID), 3.0, 1.9, "Generated\nresponse /\nclinical\njustification",
    "#EDEDED", tc="#1a1a1a", fs=11.5)

legend_elems = [
    Line2D([0], [0], marker='s', color='none', markerfacecolor=BLUE_DARK, markersize=14, label='Arm A (JSON interface)'),
    Line2D([0], [0], marker='s', color='none', markerfacecolor=ORANGE_DARK, markersize=14, label='Arm B (adapter interface)'),
]
ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, 1.06),
          ncol=2, frameon=False, fontsize=11)

ax.text(11.25, 0.55,
        "The ECG segment is a real MIT-BIH recording; the prediction values shown "
        "are illustrative, used only to demonstrate the information flow.",
        ha="center", va="center", fontsize=10.5, color=TEXT_GREY, style="italic")

fig.tight_layout()
for d in OUT_DIRS:
    fig.savefig(f"{d}/fig1_1_two_interfaces.png", dpi=200, bbox_inches="tight",
                facecolor="white")
print("saved fig1_1_two_interfaces.png to", len(OUT_DIRS), "location(s)")
