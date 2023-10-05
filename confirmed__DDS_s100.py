import numpy as np
from tqdm import tqdm

# ==================================================
# Program description
# ==================================================
#
# 

# ==================================================
# Simulation parameters
# ==================================================

timeWindow = 100
totalMCS = 500 # Total number of Monte Carlo steps per single run
runsNumber = 100
Lx = 64  # Number of rows , width
Ly = 64  # Number of columns , length
N = Lx*Ly//2 # 64000x40 right now
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

size = 1.0*Lx*Ly

# ==================================================
# Random numbers generator creating
# ==================================================

RNG = np.random.default_rng()
Latt = np.arange(Lx*Ly)
Dir = np.arange(4)
Rand = RNG.random

# ==================================================
# Beginning of the simulation
# ==================================================

for iwalk in tqdm(range(runsNumber), desc="Simulation progress"):
    #### Clearing arrays from the last run ////
    System.fill(0)
    SystemSnapshot.fill(0)
    Corr[1:] = 0
    #### Filling the lattice with particles alternatively 
    counter = 0
    for i in range(Lx):
        if counter % 2 == 0:
            System[i, ::2] = 1
            System[i, 1::2] = 0
        else:
            System[i, ::2] = 0
            System[i, 1::2] = 1
        counter += 1
    #### Beginning of a single run ////
    for istep in range(1, totalMCS):
        for moveAttempt in range(N):
            dice = RNG.integers(Lx*Ly) # Picks the random spin in the array
            X = dice // Ly
            Y = dice - X*Ly
            if System[X, Y] == 1: # We work only with positive spins		
                # Simple implementation of Periodic boundary conditions
                xPrev = Lx - 1 if X == 0 else X - 1
                xNext = 0 if X == Lx - 1 else X + 1
                yPrev = Ly - 1 if Y == 0 else Y - 1
                yNext = 0 if Y == Ly - 1 else Y + 1
                # Simulating exchange dynamics
                dice = RNG.integers(4)
                if dice == 0: # hop forward
                    temp = System[X, Y]
                    System[X, Y] = System[X, yNext]
                    System[X, yNext] = temp
                if dice == 2: # hop up
                    temp = System[X, Y]
                    System[X, Y] = System[xPrev, Y]
                    System[xPrev, Y] = temp
                if dice == 3: # hop down	
                    temp = System[X, Y]
                    System[X, Y] = System[xNext, Y]
                    System[xNext, Y] = temp
            else:
                moveAttempt -= 1

        #### Computing correlation function ////
        if istep == timeWindow: # Takes snapshoot after defined timeWindow
            SystemSnapshot = np.copy(System) # Mapping  +1/-1 spin system to the 0/1 particle system
        # Computes correlation funciton
        if istep >= timeWindow:
            Sum = np.sum(System*SystemSnapshot)
            Corr[istep] = (1.0*Sum / size) - 0.25

    #### Computing parameters after a single run  ////
    CorrTot[timeWindow:] += Corr[timeWindow:]
    #print("Completion: ", iwalk * 100 / runsNumber, "%")

test = np.sum(System)
print("Density after simulation: ", (test*1.0) / (1.0*Lx*Ly))

# ==================================================
# Simulation results output
# ==================================================

for dt in range(timeWindow, totalMCS):
    CorrTot[dt] = CorrTot[dt] / runsNumber
    print(dt, " ", 1.0*dt / (1.0*timeWindow), "	", CorrTot[dt], " ", CorrTot[dt] * timeWindow)

# ==================================================
# Memory deallocation not needed in Python
# ==================================================
