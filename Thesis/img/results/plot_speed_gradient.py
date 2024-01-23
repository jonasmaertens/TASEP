import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage

from smarttasep.Hasel import hsl2rgb
import scienceplots

plt.style.use(['science'])
plt.rcParams['font.size'] = 11
plt.rcParams['text.usetex'] = True
# plt.rcParams['figure.figsize'] = (4.5, 3.2)

fig, ax = plt.subplots(1, 2, width_ratios=[6, 1], figsize=(9, 1.5/7*9))
ax[1].yaxis.tick_right()
ax[1].yaxis.set_label_position("right")

# no spacing between subplots
plt.subplots_adjust(wspace=0)


def grad(a, x):
    return 1 - a * ((1 + a) / a) ** x + a


def inv_grad(a, y):
    return np.log((a - y + 1) / a) / np.log((1 + a) / a)


def map_grad_val_to_color(val):
    """Maps speed in the interval [0, 1] to a color in the interval red to green."""
    s = 1
    l = 0.5
    h = val / 3
    rgb = hsl2rgb(np.array([h, s, l]).reshape(1, 1, 3)) / 255
    r, g, b = rgb[0, 0, 0], rgb[0, 0, 1], rgb[0, 0, 2]
    mpl_color = matplotlib.colors.to_hex((r, g, b))
    return mpl_color


a = 100

x_arr = np.linspace(0, 1, 100)
for i in x_arr:
    ax[1].axhline(i, 0, (1.05 + (0.1) * (i - 1)) * inv_grad(a, i), color=map_grad_val_to_color(inv_grad(a, i)),
                  linewidth=1)
    ax[1].axhline(-i, 0, (1.05 + (0.1) * (i - 1)) * inv_grad(a, i), color=map_grad_val_to_color(inv_grad(a, i)),
                  linewidth=1)

ax[1].plot(x_arr, grad(a, x_arr), color="black", linewidth=3)
ax[1].plot(x_arr, -grad(a, x_arr), color="black", linewidth=3)

# include speed_grad_smarticles_{a}.png left of the plot
im = plt.imread(f"speed_grad_smarticles_{a}.png")

ax[0].set_xlim(0, 128)
ax[0].set_ylim(0, 32)
ax[0].imshow(im, extent=[0, 128, 0, 32])
# set ax[0] xlabel
ax[0].set_xlabel(f"Speed gradient linearity $a={a}$")
ax[0].set_xticks([])
ax[0].set_yticks([])




plt.xlabel("Speed")
plt.ylabel("Rel. dist. from center")
# plt.xlim(0, 1)
plt.ylim(-1, 1)
# set ticks and tick labels
plt.yticks([-1, -0.5, 0, 0.5, 1], ["1", "0.5", "0", "0.5", "1"])
plt.xticks([0, 0.5, 1], ["0", "0.5", "1"])
plt.savefig(f"speed_gradient_{a}.pdf", bbox_inches='tight')
