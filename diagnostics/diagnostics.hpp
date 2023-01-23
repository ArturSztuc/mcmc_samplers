#pragma once

// ROOT stuff
#include "TTree.h"
#include "TObjArray.h"

// std
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <iostream>

// Local libs
#include "utils/progressbar.hpp"

class Diagnostics
{

// Public member functions
public:
  Diagnostics(TTree *tree,
              std::string _logprobname = "log_prob", 
              int _burnin = 0,
              std::vector<std::string> _inactive = {"step"}
              );

  virtual ~Diagnostics(){ ClearRAM(); };

  // Get const pointer to the parameter values
  const std::vector< std::vector< double > > &GetParValues(){ return fParValues; };

  // Get a clone of 1D array of parameter values
  std::vector<double> GetParValues(int idx){ return fParValues[idx]; };
  double GetParValue(int idPar, int idEvent){ return fParValues[idPar][idEvent]; };

  std::vector<std::string> GetParNames(){return fActiveNames; };
  std::string GetParName(int idx){return fActiveNames[idx]; };

  int GetNSteps(){ return fNSteps; };
  int GetNParameters(){ return fNPar; };


  void PrintTest();

// Protected/shared member functions
protected:
  void FillRAM();
  void ClearRAM();
  void FillActiveNames();

// Data members
protected:
  TTree* fTree;
  int fNPar;
  int fNSteps;
  std::string fLogProbName;
  int fBurnIn;
  std::vector<std::string> fInactiveNames;
  std::vector<std::string> fActiveNames;
  std::vector< std::vector< double > > fParValues;
  std::vector< double > fLogLValues;
};
