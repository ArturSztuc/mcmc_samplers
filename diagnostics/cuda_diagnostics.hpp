#pragma once

// std
#include <vector>
#include <iostream>

// root
#include "TGraph.h"
#include "TStopwatch.h"

// TODO:
//  * Look into texture memory: constant memory that does not need to be
//    compiled at runtime (we would load our parameter values onto that).

class CudaDiagnostics
{
public:
  CudaDiagnostics(int _maxlag, const std::vector< std::vector< double > > &_parvals);
  virtual ~CudaDiagnostics();

  void ClearRAM();

  std::vector<TGraph*> GetAutocorrelations();

// Private function members
private:
  void ScanParameters(const std::vector< std::vector< double > > &_parvals);
  int  GetIDX(int parameter, int step);

// Private data members
private:
  int fMaxLag;
  int fNSteps;
  int fParSize;
  int fParSizeAll;

  float *fParameterValuesCPU;
  float *gParameterValuesGPU;
  float *gAutocorrelationsGPU;
  float *fAutocorrelationsCPU;

  float *fMeansCPU;
  float *gMeansGPU;
};
