import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("pgf")
plt.rcParams['text.usetex'] = True
plt.rcParams['pgf.rcfonts'] = False
plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (6, 4.3)


def rew(r, dv):
    if dv < 0.5:
        if r <= 1.5:
            return 0
        elif r > 5:
            return 0
        else:
            return -0.125 * r + 0.625
    else:
        if r > 3.5:
            return 0
        else:
            return - 0.75 / (r ** 1.3) - 0.15


# plot reward in 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

rs = np.linspace(0.85, 8, 100)
dvs = np.linspace(0, 1, 100)
rs, dvs = np.meshgrid(rs, dvs)

rews = np.empty(rs.shape)
for i in range(len(rs)):
    for j in range(len(rs[i])):
        rews[i][j] = rew(rs[i][j], dvs[i][j])

ax.plot_surface(rs, dvs, rews, cmap='viridis', edgecolor='none')
ax.set_xlabel("Distance $r$")
ax.set_ylabel("Velocity difference $\\Delta v$")
ax.set_zlabel("Reward $R(r, \\Delta v)$")

# invert axes
ax.invert_xaxis()
ax.invert_yaxis()

# set x ticks
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
# move axes labels and ticks closer to plot
ax.xaxis.labelpad = -5
ax.yaxis.labelpad = -2
ax.zaxis.labelpad = 7
ax.tick_params(axis='x', pad=-5)
ax.tick_params(axis='y', pad=-2)

# remove padding on top and left of plot while keeping padding on bottom and right
# plt.tight_layout(pad=0, rect=(0, 0, 1, 1))

plt.savefig("lane_reward_func_3d.pdf")
# plt.show()