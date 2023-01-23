#pragma once

// Eigen libraries
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/Cholesky>

// std
#include <string>
#include <iostream>
#include <vector>
#include <random>
#include <omp.h>

// ROOT
#include "TRandom3.h"
#include "TFile.h"
#include "TTree.h"

// Local libs
#include "mcmc/mcmc.hpp"
#include "model/model.hpp"
#include "utils/progressbar.hpp"


// Config for the MCMC sampler. 
// TODO:
//  * Need to move over to the new, general config class.
struct Config
{
  // Number of MCMC samples to generate
  int fNSteps;

  // Random number seed generator 
  int seed;

  // Array of step-size multipliers
  std::vector<double> fStepSize;
  // Output file name/location
  std::string fOutputFile;
  // Are we using a dense matrix?>
  bool matrixDense;
  // Are we using a diagonal matrix?>
  bool matrixDiagonal;
  // Covariance matrix, if we're using one
  Eigen::MatrixXd covariance;
};

// A simple struct holding our sample and it's logl.  
// TODO: 
//  * This is a bit pointless? No need for parameter names (outside of
//    setting the branches) either. 
//  * Maybe could have a parameter objects instead
struct Sample
{
  // Parameter values
  Eigen::VectorXd vals;
  // Parameter names
  std::vector<std::string> varNames;
  // Sample's log probability
  double logProb;
};

// Metripolis-Hastings sampler.
// The simplest (maybe apart from gibbs?) MCMC sampler out there. Can use dense
// matrices with cholesky decomposition for fine step-size tuning.
//
// TODO:
//  * Move over to the new config class...
//  * Implement eigen decomp too?
//  * Re-think how the "Sample" struct is implemented. It's dumb that we have
//    it re-implemented in each MCMC class, mostly almost identical.
//  * Maybe could re-write it in such a way that reverse-jump and
//    parallel-tempering be derived from this rather than re-implementing
//    similar/the same functions?
//  * Definitely needs more sanity checks/validations.
//template <class Model>
class MetropolisHastings : public MCMCBase 
{
  // Class data members
  private:

    // Model that we are sampling from.
    ModelBase *fModel;

    // Current and proposed MCMC sample
    Sample fCurrent;
    Sample fProposed;

    // MCMC config struct
    Config fConfig;

    // Cholesky-Decomposed LLT 
    Eigen::MatrixXd choleskyLLT;

    // Stores MCMC steps
    TTree *fOutTree;

    // Internal step-sizes to be updated at each MCMC step
    Eigen::VectorXd fStepSizes;
    //Eigen::VectorXd fUserStepSizes;

  // Class public member functions
  public:
    // Class initializer
    MetropolisHastings(ModelBase *_fModel, Config _fConfig);

    ~MetropolisHastings(){fOutFile->Close();}

    // Run MCMC
    void run_mcmc();

    TTree* get_tree(){ return fOutTree; };

  // Class private member functions
  private:
    // Proposes a new MCMC step
    void propose_step();

    // Error handling for the MCMC config
    void validate_config();

    // Compare the proposed step against current
    void metropolis_correction();

    // Initialize the output TTree
    void initialize_sampler();

};
