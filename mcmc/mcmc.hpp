#pragma once
#ifndef MCMCBASE_H
#define MCMCBASE_H

#include "TRandom3.h"
#include "TFile.h"

// Base MCMC class that all the other MCMC samplers are derived from. 
// TODO:
//  * This is stupid. Ensemble MCMC samplers work in such a different way that
//    even "nAccepted" or "nStep" are ambiguous, and in reverse-jump the "fNPars"
//    changes it's meaning. Either remove, or re-think how to use MCMCBase?
//  * Get rid of ROOT rnd and use modern C++ instead.
class MCMCBase
{
public:
  MCMCBase (){};
  virtual ~MCMCBase () {};

  virtual void run_mcmc() = 0;

protected:
    // Number of fitting parameters
    int fNPars;

    // Number of accepted MCMC steps
    int nAccepted;

    // Current MCMC step
    int fStep;

    // Random number generator. Change to boost or standard C++?
    TRandom3 *random;

    // Output file
    TFile *fOutFile;

};

#endif /* MCMC_H */
