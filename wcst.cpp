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
#include <vector>
#include <cmath>
#include "wmtk/WMtk.h"

using namespace std;

////////////////////////
//  Global variables  //
////////////////////////

// The number of dimensions to attend to. Eg. color, shape, size, number
int numberOfDimensions = 2;

// The number of features used in each dimension: number of colors, shapes, sizes, etc.
int numberOfFeatures = 3;

// The number of possible cards
int numberOfCards = pow(numberOfFeatures, numberOfDimensions);

// Random seed for repeatability
int seed = 0;

// TD values
double alpha    = 0.9;			// TD learning rate
double discount = 0.5;			// TD discount
double lambda   = 0.1;			// Eligibility trace trickle rate
double epsilon  = 0.05;			// Epsilon soft policy

int vectorSize = 1024;		// The length of the HRRs
int numberOfChunks = 1;		// Only use 1 WM slot

// Simulation variables
int numberEpisodes = 200000;
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
			numberEpisodes = 100;
		} else {
			numberOfDimensions = atoi(argv[2]);
			numberOfFeatures = atoi(argv[3]);
			numberOfCards = pow(numberOfFeatures, numberOfDimensions);
		}
		break;
	default:
		printf("Usage: wcst [seed [dimensions features]]\n");
		return 9;
	}

	//printf("big*red*ball isRule: %d\n", isARule("big*red*ball"));
	//printf("ball isRule: %d\n", isARule("ball"));


	/**** Initialize working memory ****/
	WorkingMemory wm(alpha,
					 discount,
					 lambda,
					 epsilon,
					 vectorSize,
					 numberOfChunks );
	
	/**** Set the current rule for the task ****/
	string rule = randomRule();

	/**** Create the deck of cards for the task ****/
	findCombs(0, numberOfDimensions, numberOfFeatures, "");

	/**** Get the first card ****/
	string currentCard = randomCard();

	// Initialize working memory for the task
	wm.initializeEpisode(currentCard);

	// Main loop of task
	for (int timestep = 0; timestep <= numberEpisodes; timestep++) {

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
	return dimensions[rand() % numberOfDimensions][rand() % numberOfFeatures];
}

string randomCard() {
	return cards[ rand() % numberOfCards];
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
