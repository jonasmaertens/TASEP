#include <iostream>
#include <stdlib.h>
#include <iomanip>
#include <time.h>
#include <random>
#include <cmath>
#include <fstream>

using namespace std;

// ==================================================
// Program description
// ==================================================
//
//	The goal of this program is to simulate relaxation dynamics of the KLS System. Particulary,
//	we simulate spin exchange dynamics in two dimensional lattice with periodic boundary conditiona,
//  conserved total magnetization, ferromagnetic Ising interactions, and with one spin per one lattice site and con. 
//  At the beginning up/down spins randomly distributed in the lattice. The infinite drive is then applied
//  horizontally to the right, that induces phase separation of the system (stripes emergence). 


int main() {

	// ==================================================
	// Simulation parameters
	// ==================================================

	//const int timeWindow = 200;
	const int totalMCS = 10000; // Total number of Monte Carlo steps per single run
	const int runsNumber = 10;
	const int Lx = 128;  // Number of rows , width
	const int Ly = 1000;  // Number of columns , length
	const int N = Lx*Ly / 2; // 64000x40 right now
	const int left_Border = 0; // Set boundaries for a high temperature region
	const int right_Border = 0;
	const double Temp_KLS = 0.8;  // kT/J actually
	const double Temp_DDS = 2.0;  // kT/J actually
	int temp, test, i, j, n, dt, delta, E1, E2, xPrev, xPPrev, xNext, xNNext, yPrev, yNext, yNNext, dice;
	double r, w, W;
	//const double ratio = (right_Border - left_Border) / (1.0*Ly);

	// ==================================================
	// Memory allocation
	// ==================================================
	int** System = new int*[Lx];
	double* HopProb_KLS = new double[7];
	double* HopProb_DDS = new double[7];
	double** CurrentAlong = new double*[totalMCS];
	//double** CurrentTransverse = new double*[totalMCS];
	for (i = 0; i < Lx; i++) {
		System[i] = new int[Ly];
	}
	for (dt = 0; dt < totalMCS; dt++) {
		CurrentAlong[dt] = new double[Ly];
		//CurrentTransverse[dt] = new double[Ly];
		for(i = 0; i < Ly; i++){
			CurrentAlong[dt][i] = 0;
			//CurrentTransverse[dt][i] = 0;
		}
	}

	// ==========================================================================================
	// Precompute all possible probabilities for spins exchangeand some values for optimization
	// ==========================================================================================

	for (int n = 0; n < 7; n++) {
		HopProb_KLS[n] = min(1.0, exp(1.0*(n - 3) / Temp_KLS));//right
		HopProb_DDS[n] = min(1.0, exp(1.0*(n - 3) / Temp_DDS));//right
	}

	// ==================================================
	// Random numbers generator creating
	// ==================================================

	random_device rd{};
	mt19937 RNG{ rd() };
	uniform_int_distribution<int> Latt{ 0, Lx*Ly - 1 }; // Here I mapped 2d array to 1d
	uniform_int_distribution<int> Dir{ 0, 3 }; // Choose direction
	uniform_real_distribution<double> Rand{ 0.0, 1.0 }; // a dice from 0 to 1

														// ==================================================
														// Beginning of the simulation
														// ==================================================

	for (int iwalk = 0; iwalk < runsNumber; iwalk++) {
		//// Clearing arrays from the last run ////
		for (i = 0; i < Lx; i++) {
			for (j = 0; j < Ly; j++) {
				System[i][j] = 0;
				SystemSnapshot[i][j] = 0;
			}
		}
		//// Filling the lattice with spins in a random fashion ////
		/*for (n = 0; n < N; n++) {
		dice = Latt(RNG); // Pick random cell
		int X = dice / Ly; int Y = dice - X*Ly;
		if (System[X][Y] == 0) {
		System[X][Y] = 1;
		}
		else {
		n--;
		}
		}*/
		//// Filling like checkboards
		for (i = 0; i < Lx; i += 2) {
			for (j = 0; j < Ly; j += 2) {
				System[i][j] = 1; System[i][j + 1] = 0;
				System[i + 1][j] = 0; System[i + 1][j + 1] = 1;
			}
		}

		//// Beginning of a single run ////
		for (int istep = 1; istep < totalMCS; istep++) {
			for (int moveAttempt = 0; moveAttempt < N; moveAttempt++) {
				dice = Latt(RNG); // Picks the random spin in the array
				int X = dice / Ly; int Y = dice - X*Ly; // Here 
				if (System[X][Y] == 1) { // We work only with positive spins		
										 // Simple implementation of Periodic boundary conditions
					if (X == 0) {
						xPrev = Lx - 1;
						xPPrev = Lx - 2;
					}
					if (X == 1) {
						xPrev = 0;
						xPPrev = Lx - 1;
					}
					if (X != 0 && X != 1) {
						xPrev = X - 1;
						xPPrev = X - 2;
					}
					if (X == Lx - 1) {
						xNNext = 1;
						xNext = 0;
					}
					if (X == Lx - 2) {
						xNNext = 0;
						xNext = Lx - 1;
					}
					if (X != Lx - 1 && X != Lx - 2) {
						xNNext = X + 2;
						xNext = X + 1;
					}
					yPrev = Y == 0 ? Ly - 1 : Y - 1;
					yNext = Y == Ly - 1 ? 0 : Y + 1;

					// Simulating exchange dynamics
					dice = Dir(RNG);
					if (dice == 0) { // hop right
						temp = System[X][Y];
						System[X][Y] = System[X][yNext];
						System[X][yNext] = temp;
						CurrentAlong[istep][Y]++;
					}
					if (dice == 2 && System[xPrev][Y] == 0) {// hop up
						E1 = System[xNext][Y] + System[X][yPrev] + System[X][yNext];
						E2 = System[xPPrev][Y] + System[xPrev][yPrev] + System[xPrev][yNext];
						delta = E2 - E1; // energy change due particle hop
						if (Y >= left_Border && Y < right_Border) { // different temperatures for different regions
							W = HopProb_DDS[delta + 3];
						}
						else {
							W = HopProb_KLS[delta + 3];
						}
						w = Rand(RNG);
						if (w < W) {
							temp = System[X][Y];
							System[X][Y] = System[xPrev][Y];
							System[xPrev][Y] = temp;
						//	CurrentTransverse[dt][Y]++;
						}
					}
					if (dice == 3 && System[xNext][Y] == 0) {// moving spin down by exchange	
						E1 = System[xPrev][Y] + System[X][yPrev] + System[X][yNext];
						E2 = System[xNNext][Y] + System[xNext][yPrev] + System[xNext][yNext];
						delta = E2 - E1; // energy change due particle hop
						if (Y >= left_Border && Y < right_Border) { // different temperatures for different regions
							W = HopProb_DDS[delta + 3];
						}
						else {
							W = HopProb_KLS[delta + 3];
						}
						w = Rand(RNG);
						if (w < W) {
							temp = System[X][Y];
							System[X][Y] = System[xNext][Y];
							System[xNext][Y] = temp;
						//	CurrentTransverse[dt][Y]--;
						}
					}
				}
			}
		}
	}
	// ==================================================
	// Record output
	// ==================================================

	for(dt = 0; dt < totalMCS; dt++){
		cout << dt << "  ";
		for(i = 0; i < Ly; i++){
			CurrentAlong[dt][i] = CurrentAlong[dt][i]/(runsNumber*1.0);
			//CurrentTransverse[dt][i] = CurrentTransverse[dt][i]/(runsNumber*1.0);
			cout<< CurrentAlong[dt][i] << "  ";
			//cout<< CurrentTransverse[dt][i] << "  ";
		}
		cout << endl;
	}
	
	// ==================================================
	// Memory deallocation
	// ==================================================

	/*delete[] System;
	delete[] SystemSnapshot;*/
	delete[] Corr;
	delete[] CorrTot;
	delete[] Corr_DDS;
	delete[] Corr_KLS;
	delete[] CorrTot_DDS;
	delete[] CorrTot_KLS;

	return 0;

}
