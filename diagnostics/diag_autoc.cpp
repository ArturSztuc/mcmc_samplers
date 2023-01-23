#include "diag_autoc.hpp"
#include "omp.h"

// TODO:
//  * Figure out how to do this on GPU/CUDA

DiagAutocorrelations::DiagAutocorrelations(TTree *tree)
  : Diagnostics(tree)
{
}

TGraph *DiagAutocorrelations::GetAutocorrelation(int parameter, int maxLag)
{
  TGraph *ret = new TGraph(maxLag);
  ret->SetTitle(";Lag;Autocorrelation");

  std::vector< double > numerator(maxLag, 0);
  std::vector< double > diff(fNSteps, 0);
  std::vector< double > autocorrelations(maxLag, 0);

  double denominator = 0;

  // Get the mean
  double mean = 0;
  for (int i = 0; i < fNSteps; ++i)
    mean += fParValues[parameter][i];
  mean /= fNSteps;

  // Get the denominator
  for (int i = 0; i < fNSteps; ++i) {
    diff[i] = fParValues[parameter][i] - mean;
    denominator += diff[i]*diff[i];
  }

  // Loop over lags
#pragma omp parallel for
  for (int iLag = 0; iLag < maxLag; ++iLag) {
    // Loop over the entries
    for (int i = 0; i < fNSteps; ++i) {
      // Calculate the numerator first: lag
      if(i < fNSteps - iLag){
        double lagTerm = fParValues[parameter][i+iLag] - mean;
        double product = diff[i]*lagTerm;
        numerator[iLag] += product;
      }
    }
    autocorrelations[iLag] = numerator[iLag]/denominator;
    //printf("Par: %i, Lag: %i, Autocor: %f\n", parameter, iLag, autocorrelations[iLag]);
  }
  for (int iLag = 0; iLag < maxLag; ++iLag) 
    ret->SetPoint(iLag, iLag, autocorrelations[iLag]);

  return ret;
}
