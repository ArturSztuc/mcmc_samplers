#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "model/model.hpp"
#include "model/model_covariance.hpp"
#include "mcmc/mcmc_methastings.hpp"

#include "TMatrixDSym.h"
#include "TFile.h"

int main(int argc, char *argv[])
{

  // Use one of the T2K covariance matrices for tests
  TFile file_cov("/home/artur/work/t2k/MaCh3/inputs/BANFF_DataFit_2018_final_180712_sk.root");
  TMatrixDSym *cov = (TMatrixDSym*)file_cov.Get("postfit_cov");

  // Initialise the model
  ModelBase *mod = new ModelCovariance(cov);

  // Initialise the step-sizes.
  std::vector<double> step_sizes;
  for (int i = 0; i < cov->GetNrows(); ++i) {
    // Completely unoptimised, and not using matrix for step-size tuning
    //step_sizes.push_back(0.0025); 

    // Additionally optimised with the diagonal matrix
    //step_sizes.push_back(0.7); 

    // Additionally optimised with the dense matrix
    step_sizes.push_back(0.3); 
  }

  // Extract the covariance matrix for step-size tuning
  // There must be a nicer way of doing this
  double *tmp = cov->GetMatrixArray();
  Eigen::MatrixXd covariance = Eigen::Map<Eigen::MatrixXd>(tmp, 
                                                           cov->GetNrows(),
                                                           cov->GetNcols());

  // Initialise Metropolis-Hastings config. Should move to the general config
  // class instead...
  Config config;
  config.fNSteps = 20000;
  config.fStepSize = step_sizes;
  config.fOutputFile = "out.root";
  // Cholesky decomp
  config.matrixDense = true;
  // Diagonal matrix for tuning
  config.matrixDiagonal = false;
  // Not needed if the two options above are false
  config.covariance = covariance;

  // Initialise the sampler
  MetropolisHastings sampler(mod, config);
  sampler.run_mcmc();
  
  return 0;
}
