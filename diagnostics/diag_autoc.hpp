#pragma once

// ROOT things 
#include "TTree.h"
#include "TGraph.h"

// std
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <iostream>

// Local
#include "diagnostics.hpp"

class DiagAutocorrelations : public Diagnostics
{

public:
  DiagAutocorrelations (TTree* tree);

  TGraph* GetAutocorrelation(int parameter, int maxLag = 1000);
};

