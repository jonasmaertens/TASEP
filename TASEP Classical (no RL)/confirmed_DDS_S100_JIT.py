import numpy as np
from numba import njit

# ==================================================
# Program description
# ==================================================
#
#

# ==================================================
# Simulation parameters
# ==================================================

timeWindow = 1
totalMCS = 1000  # Total number of Monte Carlo steps per single run
runsNumber = 100  # Number of runs to average over
Lx = 128  # Number of rows , width
Ly = 128  # Number of columns , length
N = Lx * Ly // 2  # 64000x40 right now
r = 0

# ==================================================
# Memory allocation
# ==================================================

System = np.zeros((Lx, Ly), dtype=int)
SystemSnapshot = np.zeros((Lx, Ly), dtype=int)
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
def simulate(System, SystemSnapshot, Corr, CorrTot):
    total_forward, total_up, total_down = 0, 0, 0
    for iwalk in range(runsNumber):
        if iwalk % (runsNumber / 100) == 0:
            print(f"Run number: {iwalk}/{runsNumber}\r")
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
                        if System[X, Y] == 0:
                            total_forward += 1
                    elif dice == 2:  # hop up
                        System[X, Y], System[xPrev, Y] = System[xPrev, Y], System[X, Y]
                        total_up += 1
                    elif dice == 3:  # hop down
                        System[X, Y], System[xNext, Y] = System[xNext, Y], System[X, Y]
                        total_down += 1
                else:
                    moveAttempt -= 1
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
        CorrTot[timeWindow:] += Corr[timeWindow:]
    curr_forward_backward = total_forward / runsNumber / totalMCS / N
    curr_up_down = (total_up - total_down) / runsNumber / totalMCS / N
    return System, SystemSnapshot, Corr, CorrTot, curr_forward_backward, curr_up_down


System, SystemSnapshot, Corr, CorrTot, curr_forward_backward, curr_up_down = simulate(
    System, SystemSnapshot, Corr, CorrTot
)
test = np.sum(System)
print("Density after simulation: ", (test) / (Lx * Ly))
print("Current Forward: ", curr_forward_backward)
print("Current Up/Down: ", curr_up_down)

# ==================================================
# save simulation results
# ==================================================

CorrTot[timeWindow:] = CorrTot[timeWindow:] / runsNumber

with open("sim_results_py.csv", "w") as f:
    f.write("t,C,C*t\n")
    for t in range(timeWindow, totalMCS):
        f.write(f"{t},{CorrTot[t]},{CorrTot[t] * t}\n")
