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
// 


int main() {

	// ==================================================
	// Simulation parameters
	// ==================================================

	int timeWindow = 1;
	int totalMCS = 1000; // Total number of Monte Carlo steps per single run
	int runsNumber = 100;
	int Lx = 128;  // Number of rows , width
	int Ly = 128;  // Number of columns , length
	int N = Lx*Ly/2; // 64000x40 right now
	int temp, test, i, j, n, dt, xPrev, xPPrev, xNext, xNNext, yPrev, yNext, dice;
	double r;

	// ==================================================
	// Memory allocation
	// ==================================================

	int** System = new int*[Lx];
	int** SystemSnapshot = new int*[Lx];
	// Correlation function for the whole system
	double* Corr = new double[totalMCS];
	double* CorrTot = new double[totalMCS];
	for (i = 0; i < Lx; i++) {
		System[i] = new int[Ly];
		SystemSnapshot[i] = new int[Ly];
	}
	for (dt = 0; dt < totalMCS; dt++) {
		Corr[dt] = 0;
		CorrTot[dt] = 0;
	}

	// ==========================================================================================
	// Precompute all possible probabilities for spins exchangeand some values for optimization
	// ==========================================================================================

	double size = 1.0*Lx*Ly;

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
		for (dt = 1; dt < totalMCS; dt++) {
			Corr[dt] = 0;
		}
		//// Filling the lattice with particles in a random or checkboard fashion ////
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
		//// Filling the lattice with particles alternatively 
		int counter =0;
		for ( i = 0; i < Lx; i++) {
			if (counter % 2 == 0) {
				for ( j = 0; j < Ly; j += 2) {
					System[i][j] = 1;
				}
				for (j = 1; j < Ly; j += 2) {
					System[i][j] = 0;
				}
			}
			else {
				for (j = 0; j < Ly; j += 2) {
					System[i][j] = 0;
				}
				for (j = 1; j < Ly; j += 2) {
					System[i][j] = 1;
				}
			}
			counter++;
		}
		//// Beginning of a single run ////
		for (int istep = 1; istep < totalMCS; istep++) {
			for (int moveAttempt = 0; moveAttempt < N; moveAttempt++) {
				dice = Latt(RNG); // Picks the random spin in the array
				int X = dice / Ly; int Y = dice - X*Ly; // Here 
				if (System[X][Y] == 1) { // We work only with positive spins		
					// Simple implementation of Periodic boundary conditions
					xPrev = X == 0 ? Lx - 1 : X - 1;
					xNext = X == Lx -1 ? 0 : X + 1;
					yPrev = Y == 0 ? Ly - 1 : Y - 1;
					yNext = Y == Ly - 1 ? 0 : Y + 1;
					// Simulating exchange dynamics
					dice = Dir(RNG);
					if (dice == 0) { // hop forward
						temp = System[X][Y];
						System[X][Y] = System[X][yNext];
						System[X][yNext] = temp;
					}
					if (dice == 2) {// hop up
						temp = System[X][Y];
						System[X][Y] = System[xPrev][Y];
						System[xPrev][Y] = temp;
					}
					if (dice == 3) {// hop down	
						temp = System[X][Y];
						System[X][Y] = System[xNext][Y];
						System[xNext][Y] = temp;
					}
				}
				else{
					moveAttempt--;
				}
			}

			//// Computing correlation function ////
			if (istep == timeWindow) { // Takes snapshoot after defined timeWindow
				for (i = 0; i < Lx; i++) {
					for (j = 0; j < Ly; j++) {
						SystemSnapshot[i][j] = System[i][j]; // Mapping  +1/-1 spin system to the 0/1 particle system
					}
				}
			}
			// Computes correlation funciton
			if (istep >= timeWindow) {
				int Sum = 0;
				for (i = 0; i < Lx; i++) {
					for (j = 0; j < Ly; j++) {
						Sum += System[i][j] * SystemSnapshot[i][j];
					}
				}
				Corr[istep] = (1.0*Sum / size) - 0.25;
			}
		}

		//// Computing parameters after a single run  ////
		for (dt = timeWindow; dt < totalMCS; dt++) {
			CorrTot[dt] += Corr[dt];
		}
		//cout << "Completion: " << iwalk * 100 / runsNumber << "%" << endl;
	}

	test = 0;
	for (i = 0; i < Lx; i++) {
		for (j = 0; j < Ly; j++) {
			test += System[i][j];
		}
	}
	cout << "Density after simulation: " << (test*1.0) / (1.0*Lx*Ly) << endl;

	// ==================================================
	// Simulation results output
	// ==================================================

	ofstream file;
	file.open("sim_results_cpp.csv");
	//cout << "MCS" << " " << "t/s" << " " << "S(s,t)" << " " << "S(s,t)*s^0.5" << " " << "S(s,t)*s" << " " << "S_DDS(s,t)" << " " << "S_DDS(s,t)*s^0.5" << " " << "S_DDS(s,t)*s" << " " << "S_KLS(s,t)" << " " << "S_KLS(s,t)*s^0.5" << " " << "S_KLS(s,t)*s" << endl;
	file << "t,C,C*t" << endl;
	for (dt = timeWindow; dt < totalMCS; dt++) {
		CorrTot[dt] = CorrTot[dt] / runsNumber;
		file << dt << "," << CorrTot[dt] << "," << CorrTot[dt] * dt << endl;
	}
	file.close();

	// ==================================================
	// Memory deallocation
	// ==================================================

	for(i = 0; i< Lx; i++){
		delete System[i];
		delete SystemSnapshot[i];
	}

	delete[] System;
	delete[] SystemSnapshot;
	delete[] Corr;
	delete[] CorrTot;

	system("PAUSE");
	return 0;

}
