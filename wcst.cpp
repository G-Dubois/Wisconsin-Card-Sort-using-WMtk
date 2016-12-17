// Wisconsin Card Sort Taks
// Demo Application for the Working Memory toolkit
// Copyright (C) 2016, Grayson M. Dubois and Joshua L. Phillips
// Department of Computer Science, Middle Tennessee State University

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have recieved a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

//
// wcst.cpp
//
// Provides source for for Wisconsin Card Sort simulation.
//
// Original code by Grayson M. Dubois, based on examples from Joshua L. 
// Phillips.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "wmtk/WMtk.h"
#include "wmtk/hrr/hrrengine.h"
#include "wmtk/hrr/hrrOperators.h"

using namespace std;

////////////////////////
//  Global variables  //
////////////////////////

// Random seed for repeatability
int seed = 0;

// The number of dimensions to attend to. Eg. color, shape, size, number
int ndims = 2;

// The number of features used in each dimension: number of colors, shapes, sizes, etc.
int nfeatures = 3;

// The number of possible cards
int ncards = pow(nfeatures, ndims);

// The length of the HRRs
int n = 64;
int numberOfChunks = 1;		// Only use 1 WM slot

// TD values
double alpha    = 0.9;			// TD learning rate
double discount = 0.5;			// TD discount
double lambda   = 0.1;			// Eligibility trace trickle rate
double epsilon  = 0.05;			// Epsilon soft policy

// Simulation variables
int nsteps = 1000000;
int numberOfCorrectTries = 0;
int numberOfIncorrectTries = 0;

// Vector containing all possible cards
vector<string> cards;

string colors[] = {"red", "green", "blue", "yellow"};
string shapes[] = {"square", "circle", "triangle", "oval"};
string sizes [] = {"small", "medium", "large", "giant"};

string* dimensions[] = {colors, shapes, sizes};

bool debug = false;

void printAllCombinations(string* array[], int D, int N);
void findCombs(int, int, int, string);
string randomRule();
string randomCard();
bool cardMatchesRule(string, string);
bool isARule(string);

int main(int argc, char* argv[]) {

	/**** Get input from command line arguments ****/
	switch (argc) {
	case 2:
		seed = atoi(argv[1]);
		srand(seed);
		break;
	case 4:
		seed = atoi(argv[1]);
		srand(seed);
		if (argv[2] == "debug"){
			debug = true;
			nsteps = 100;
		} else {
			ndims = atoi(argv[2]);
			nfeatures = atoi(argv[3]);
			ncards = pow(nfeatures, ndims);
		}
		break;
	default:
		printf("Usage: wcst [seed [dimensions features]]\n");
		return 9;
	}


	/**** Initialize working memory ****/
	WorkingMemory wm(alpha,
			 discount,
			 lambda,
			 epsilon,
			 n,
			 numberOfChunks );
	
	/**** Set the current rule for the task ****/
	string rule = randomRule();

	/**** Create the deck of cards for the task ****/
	findCombs(0, ndims, nfeatures, "");

	/**** Insert all cards into working memory and print their hrrs ****/
	ofstream fout;
	fout.open("temp_hrrs_c.dat");
	printf("Number of cards: %d\t ncards: %d\n", (int)cards.size(), ncards);
	for (string card : cards) {
		cout << card << "\n";
		wm.hrrengine.query(card);
	}
	for (pair<string, HRR> concept : wm.hrrengine.conceptMemory) {
		if (isARule(concept.first) && concept.first != "I") {
			cout << "Rule: " << concept.first << "\n";
			fout << concept.second << "\n";
		}
	}
	fout.close();

	fout.open("temp_weights_c.dat");
	fout << wm.weights << "\n";
	fout.close();

	/**** Set up percept sequence ****/
	int percepts[nsteps][2];
	fout.open("temp_percepts_c.dat");
	for (int i = 0; i < nsteps; i++) {
		for (int d = 0; d < ndims; d++) {
			percepts[i][d] = rand()%nfeatures;
			fout << percepts[i][d] << " ";
		}
		fout << "\n";
	}
	fout.close();

	/**** Get the first card ****/
	string currentCard = randomCard();

	// Initialize working memory for the task
	wm.initializeEpisode(currentCard);

	// Main loop of task
	for (int timestep = 0; timestep <= nsteps; timestep++) {

		bool cardIsAMatch = cardMatchesRule(currentCard, rule);
		bool chooseCorrect;

		// Choose the pile to place card in
		if (!isARule(wm.queryWorkingMemory(0)) || wm.queryWorkingMemory(0) == "I") {
			chooseCorrect = (bool) (rand() % 2);
		} else {
			chooseCorrect = cardMatchesRule(currentCard, wm.queryWorkingMemory(0));
		}

		if (debug) {
			printf("Rule: %s\n", rule.c_str());
			printf("Card: %s\n", currentCard.c_str());
			printf("Card Matches Rule: %d\n\n", cardIsAMatch);
			printf("WM Contents: %s\n", wm.queryWorkingMemory(0).c_str());
			printf("Chosen pile: %d\n\n\n", chooseCorrect);
		}

		// Check for correct move
		if (cardIsAMatch == chooseCorrect) {
			numberOfCorrectTries++;
			wm.step(currentCard, 1.0);
		} else {
			numberOfIncorrectTries++;
			numberOfCorrectTries = 0;
			wm.step(currentCard, 0.0);
		}

		// Change the rule after 100 correct tries
		if ( numberOfCorrectTries > 100 ) {
			numberOfCorrectTries = 0;

			string newRule = randomRule();
			while (newRule == rule) {
				newRule = randomRule();
			}
			rule = newRule;

			printf("Timestep: %d - new rule [%s] - incorrect tries [%d]\n", timestep, rule.c_str(), numberOfIncorrectTries);
			numberOfIncorrectTries = 0;
		}

		// Draw a new card
		string newCard = randomCard();
		while (newCard == currentCard) {
			newCard = randomCard();
		}
		currentCard = newCard;

		if (timestep % 1000 == 0) {
			printf("Timestep: %d - failures [%d] - successes [%d]\n", timestep, numberOfIncorrectTries, numberOfCorrectTries);
		}
	}

	return 0;
}

void findCombs(int d, int D, int F, string combSoFar) {

	for (int i = 0; i < F; i++) {

		if (d >= D-1) {
			cards.push_back(combSoFar + ((d == 0) ? "" : "*") + dimensions[d][i]);
		} else {
			findCombs(d+1, D, F, combSoFar + ((d == 0) ? "" : "*") + dimensions[d][i]);
		}
	}

	return;
}

string randomRule() {
	return dimensions[rand() % ndims][rand() % nfeatures];
}

string randomCard() {
	return cards[ rand() % ncards];
}

bool cardMatchesRule(string card, string rule) {

	bool match = false;

	vector<string> featuresInCard;

	int tokenBegin = 0;
	int tokenEnd = card.find('*');

	if ( rule == card.substr(tokenBegin, tokenEnd) ) {
		match = true;
	}

	while (tokenEnd <= card.length()) {
		tokenBegin = tokenEnd + 1;
		tokenEnd = card.find('*', tokenBegin);

		if (rule == card.substr(tokenBegin, tokenEnd)) {
			match = true;
		}
	}

	return match;
}

bool isARule(string concept) {

	if (concept.find("*") == string::npos) {
		return true;
	}

	return false;
}
