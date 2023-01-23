#pragma once

// ROOT things 
#include "TTree.h"
#include "TObjArray.h"
#include "TGraph.h"
#include "TH1D.h"
#include "TH2D.h"

// std
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <iostream>

// Local
#include "diagnostics.hpp"

struct PosteriorConfig
{
  // Burn-in
  int burnin = 0;
  // Number of bins
  int nbinsx = 100;
  int nbinsy = 100;

  // Min and max
  double minx = 0;
  double maxx = 0;
  double miny = 0;
  double maxy = 0;

  // Titles <3
  std::string titles = ";;;";

  // Apply smoothing?
  bool smooth = false;

  // Normalize?
  bool normalize = false;

} fDefaultPosteriorConfig;

class DiagPosterior: Diagnostics
{
public:
  DiagPosterior (TTree *tree);

  // 1D intervals
  TH1D* GetPosteriorProbability(int parameter, PosteriorConfig cfg = fDefaultPosteriorConfig);
  TH1D* GetInterval(TH1D* th, double interval); 

  // 2D Intervals
  TH2D* GetPosteriorProbability(int parameter1, int parameter2);


};
