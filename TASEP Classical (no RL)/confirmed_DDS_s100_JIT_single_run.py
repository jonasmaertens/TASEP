import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==================================================
# Program description
# ==================================================
#
#

# ==================================================
# Simulation parameters
# ==================================================

timeWindow = 100
totalMCS = 30  # Total number of Monte Carlo steps per single run
Lx = 16  # Number of rows , width
Ly = 128  # Number of columns , length
N = Lx * Ly // 2  # 64000x40 right now
r = 0

# ==================================================
# Memory allocation
# ==================================================

System = np.zeros((Lx, Ly), dtype=int)
SystemSnapshot = np.zeros((Lx, Ly), dtype=int)
memory = np.zeros((Lx, Ly, totalMCS * N), dtype=int)
# Correlation function for the whole system
Corr = np.zeros(totalMCS)
CorrTot = np.zeros(totalMCS)

# ==========================================================================================
# Precompute all possible probabilities for spins exchangeand some values for optimization
# ==========================================================================================

size = Lx * Ly

# ==================================================
# Random numbers generator creating
# ==================================================

Latt = np.arange(Lx * Ly)
Dir = np.arange(4)

# ==================================================
# Beginning of the simulation
# ==================================================


@njit
def simulate(System, SystemSnapshot, Corr, CorrTot, memory):
    #### Clearing arrays from the last run ////
    System.fill(0)
    SystemSnapshot.fill(0)
    Corr[1:] = 0
    #### Filling the lattice with particles alternatively
    System[::2, ::2] = 1
    System[1::2, 1::2] = 1
    #### Beginning of a single run ////
    for istep in range(1, totalMCS):
        moveAttempt = 1
        while moveAttempt <= N:
            dice = np.random.randint(Lx * Ly)  # Picks the random spin in the array
            X = dice // Ly
            Y = dice - X * Ly
            if System[X, Y] == 1:  # We work only with positive spins
                # Simple implementation of Periodic boundary conditions
                xPrev = Lx - 1 if X == 0 else X - 1
                xNext = 0 if X == Lx - 1 else X + 1
                yPrev = Ly - 1 if Y == 0 else Y - 1
                yNext = 0 if Y == Ly - 1 else Y + 1
                # Simulating exchange dynamics
                dice = np.random.randint(4)
                if dice == 0:  # hop forward
                    System[X, Y], System[X, yNext] = System[X, yNext], System[X, Y]
                elif dice == 2:  # hop up
                    System[X, Y], System[xPrev, Y] = System[xPrev, Y], System[X, Y]
                elif dice == 3:  # hop down
                    System[X, Y], System[xNext, Y] = System[xNext, Y], System[X, Y]
            else:
                moveAttempt -= 1
            #### Saving the system state ////
            memory[:, :, (istep - 1) * N + moveAttempt] = System
            moveAttempt += 1

        #### Computing correlation function ////
        if istep == timeWindow:  # Takes snapshoot after defined timeWindow
            SystemSnapshot = np.copy(
                System
            )  # Mapping  +1/-1 spin system to the 0/1 particle system
        # Computes correlation funciton
        if istep >= timeWindow:
            Sum = np.sum(System * SystemSnapshot)
            Corr[istep] = (Sum / size) - 0.25
    #### Computing parameters after a single run  ////

    return System, SystemSnapshot, Corr, CorrTot, memory


System, SystemSnapshot, Corr, CorrTot, memory = simulate(
    System, SystemSnapshot, Corr, CorrTot, memory
)
test = np.sum(System)
print("Density after simulation: ", (test) / (Lx * Ly))

# ==================================================
# Simulation results output
# ==================================================

with open("sim_results.csv", "w") as f:
    f.write("dt,C,C*dt\n")
    for dt in range(timeWindow, totalMCS):
        f.write(f"{dt},{Corr[dt]},{Corr[dt]*dt}\n")

fig = plt.figure(figsize=(64, 8))
im = plt.imshow(memory[:, :, 1], interpolation="none", aspect="auto", vmin=0, vmax=1)

def animate_func(i):
    im.set_array(memory[:, :, i*20])
    return [im]


fps = 60

anim = animation.FuncAnimation(
    fig,
    animate_func,
    frames=totalMCS * N // 20,
    interval=1000 / fps,  # in ms
)

print("Saving animation...")
anim.save("test_anim.mp4", fps=fps, extra_args=["-vcodec", "libx264"])
print("Done!")
