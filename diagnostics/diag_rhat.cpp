#include "diag_rhat.hpp"
#include <numeric>

DiagRhat::DiagRhat(std::vector<Diagnostics*> &_diagnostics)
  : fDiagnostics(_diagnostics)
{
  nChains = static_cast<int>(fDiagnostics.size());
  assert(nChains > 1 && "More than one chain needed for RHat calculation!");

  nSteps = fDiagnostics.first()->GetNSteps();
  nParameters = fDiagnostics.first->GetNParameters();
  for(auto diag : fDiagnostics){
    assert(nSteps== diag->GetNSteps() && 
           "Each chain must have equal number of MCMC steps for RHat calculation!");
    assert(nParameters == diag->GetNParameters() && 
           "Each chain must have equal number of parameters for RHat calculation!");
  }
}

void DiagRhat::CalculateMeanPerChain()
{
  // Omp this bitch!
#pragma omp parallel for
  for(int iChain = 0; iChain < nChains; ++iChain){
    for(int iPar = 0; iPar < nParameters; ++iPar){
      double mean = 0;
      for(int iStep = 0; iStep < nSteps; ++iStep){
        mean += fDiagnostics[iChain]->GetParameterValue(iPar, iStep);
      }
      mean /= nSteps;
    }
  }
}



void DiagRhat::CalculateRhatSplit(int idx)
{
  std::vector<std::vector<double>> fPars;

  // Get the parameter values
  for(auto diag : fDiagnostics){
    std::vector<double> tmp = diag->GetParValues(idx);
    fPars.push_back(tmp);
  }

  // Get means for each chain
  std::vector<double> fMeansPerChain;
  double fMean = 0;
  double fNPars= 0;
  for(auto par : fPars){
    double mean = std::accumulate(par.begin(), par.end(), 0); 
    fMean += mean;
    fNPars += par.size();
    mean /= par.size();
    fMeansPerChain.push_back(mean);
  }
  fMean /= fNPars;

  double B = 0;
  for(auto mean : fMeansPerChain){
    double diff = mean - fMean;
    double diffSq = diff * diff;
    B += diffSq;
  }
}
