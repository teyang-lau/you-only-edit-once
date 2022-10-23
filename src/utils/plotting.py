import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


def gaussian_smooth(x, y, grid, sd):
    weights = np.transpose([stats.norm.pdf(grid, m, sd) for m in x])
    weights = weights / weights.sum(0)
    return (weights * y).sum(1)


def streamgraph(y):

    x = np.array(range(1, len(y[0]) + 1))
    grid = np.linspace(1, len(y[0]) + 1, num=500)
    y_smoothed = [gaussian_smooth(x, y_, grid, 1.5) for y_ in y]

    COLORS = ["#66a3ff", "#ffa64d", "#33cc33", "#ff6666", "#a64dff"]
    LABELS = ["Fish", "Coral", "Turtle", "Shark", "Manta Ray"]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title("Number of objects detected across your diving video", size=12)
    ax.set_xlabel("Frame", size=8)
    ax.stackplot(grid, y_smoothed, colors=COLORS, labels=LABELS, baseline="sym")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.35),
        ncols=5,
        frameon=False,
        fontsize=8,
    )
    ax.axhline(0, color="gray", ls="--")

    return fig
