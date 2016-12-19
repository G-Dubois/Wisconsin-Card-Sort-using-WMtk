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
#include <queue>
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

double default_reward = 0.0;
double correct_reward = 1.0;

// Simulation variables
int nsteps = 5;
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
string changeRule(string oldRule);
string changeRule(vector<int>);
string drawCard(vector<int>);
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
		if (strncmp(argv[2], "debug", 6)){
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

	wm.WMdebug = true;

	/**** Create the deck of cards for the task ****/
	findCombs(0, ndims, nfeatures, "");

	/**** Insert all cards into working memory and print their hrrs ****/
	ofstream fout;
	printf("Saving HRR data...");
	fout.open("temp_hrrs_c.dat");
	for (string card : cards) {
		wm.hrrengine.query(card);
	}
	for (pair<string, HRR> concept : wm.hrrengine.conceptMemory) {
		if (isARule(concept.first) && concept.first != "I") {
			fout << concept.second << "\n";
		}
	}
	fout.close();
	printf("done!\n");

	printf("Saving weight data...");
	fout.open("temp_weights_c.dat");
	fout << wm.weights << "\n";
	fout.close();
	printf("done!\n");

	/**** Set up percept sequence ****/
	printf("Saving percept data...");
	vector<vector<int>> percepts;
	fout.open("temp_percepts_c.dat");
	for (int i = 0; i < nsteps; i++) {
		vector<int> percept;
		for (int d = 0; d < ndims; d++) {
			percept.push_back(rand()%nfeatures);
			fout << percept[d] << " ";
		}
		percepts.push_back(percept);
		fout << "\n";
	}
	fout.close();
	printf("done!\n");

	/**** Set up random rules ****/
	printf("Saving rule data...");
	queue<vector<int>> rules;
	fout.open("temp_rules_c.dat");
	for (int i = 0; i < nsteps; i++) {
		vector<int> rule;
		rule.push_back(rand() % nfeatures);
		rule.push_back(rand() % ndims);
		fout << rule[0] << " " << rule[1] << "\n";
		rules.push(rule);
	}
	fout.close();
	printf("done!\n");
	
	/**** Set the current rule for the task ****/
	printf("Rule change: [%d %d]\n", rules.front()[0], rules.front()[1]);
	string rule = changeRule(rules.front());
	rules.pop();

	/**** Get the first card ****/
	string currentCard = drawCard(percepts[0]);

	// Initialize working memory for the task
	wm.initializeEpisode(currentCard);

	double reward = 0;

	// Main loop of task
	for (int timestep = 1; timestep < nsteps; timestep++) {

		wm.step(currentCard, reward);

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
			reward = correct_reward;
			if (debug) {
				printf("Loaded correct rule...\n");
			}
		} else {
			numberOfIncorrectTries++;
			numberOfCorrectTries = 0;
			reward = default_reward;
		}

		//if (numberOfCorrectTries >= 90) {
			debug = true;
		//} else {
		//	debug = false;
		//}

		// Change the rule after 100 correct tries
		if ( numberOfCorrectTries > 100 ) {
			numberOfCorrectTries = 0;

			vector<int> r = rules.front();
			rule = changeRule(r);
			rules.pop();

			printf("Timestep: %d - new rule [%d %d] ""\n", timestep, r[0], r[1]);
			numberOfIncorrectTries = 0;
		}

		// Draw a new card
		currentCard = drawCard(percepts[timestep]);

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
string changeRule(string oldRule) {
	int d, f;
	do {
		d = rand() % ndims;
		f = rand() % nfeatures;
	} while (dimensions[d][f] == oldRule);
	printf("Rule change: [%d %d]\n", d, f);
	return dimensions[d][f];
}

string changeRule(vector<int> rule) {
	return dimensions[rule[1]][rule[0]];
}

string randomCard() {
	return cards[ rand() % ncards];
}

string drawCard(vector<int> features) {
	string card = dimensions[0][features[0]];
	for (int i = 1; i < features.size(); i++) {
		card += "*" + dimensions[i][features[i]];
	}

	return card;
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
