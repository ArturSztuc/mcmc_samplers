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
#include <numeric>

// Local
#include "diagnostics.hpp"

class DiagRhat
{

public:
  DiagRhat(std::vector<Diagnostics*> _diagnostis);

private:
  void CalculateRhatSplit(int idx);
  void CalculateMeanPerChain();

private:
  std::vector<Diagnostics*> &fDiagnostics;

  // Number of MCMC chains
  int nChains;
  // Number of MCMC steps in each chain. All chains must have equal lengths for RHat!
  int nSteps;
  // Number of parameters in each chain. Each chain must have equal number of parameters!
  int nParameters;

  // Variance between-chains for each parameter
  std::vector<double> fVarBetween; 
  // Variance within-chains for each parameter
  std::vector<double> fVarWithin;

  // Mean witihin each chain for each parameter
  std::vector<std::vector<double>> fMeanWithin;
  // Total mean for each parameter
  std::vector<double> fMeanTotal;


};

