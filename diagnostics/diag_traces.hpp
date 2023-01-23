#pragma once

// ROOT things 
#include "TTree.h"
#include "TObjArray.h"
#include "TGraph.h"
#include "TH1D.h"

// std
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <iostream>

// Local
#include "diagnostics.hpp"

class DiagTraces : Diagnostics
{
public:
  DiagTraces (TTree *tree);

  TGraph* GetTrace(int parameter);
};
