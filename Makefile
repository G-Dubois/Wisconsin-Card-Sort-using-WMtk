all:	wcst
	
wcst:	wcst.cpp wmtk/CriticNetwork.cpp wmtk/WorkingMemory.cpp wmtk/hrr/hrrengine.cpp wmtk/hrr/hrrOperators.cpp
	g++ -std=c++11 -g wcst.cpp wmtk/CriticNetwork.cpp wmtk/WorkingMemory.cpp wmtk/hrr/hrrengine.cpp wmtk/hrr/hrrOperators.cpp -lgsl -lblas -o wcst
