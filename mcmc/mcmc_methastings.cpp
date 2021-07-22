#include "mcmc_methastings.hpp"

//namespace mcmc
//{

MetropolisHastings::MetropolisHastings(ModelBase *_fModel, Config _fConfig)
  : MCMCBase(), fModel(_fModel), fConfig(_fConfig)
{
  // Initialize the sampler
  initialize_sampler();
};

// Make sure the config will actually work!
//template <class Model>
void MetropolisHastings::validate_config()
{
  // The number of MCMC steps must be a positive integer
  assert(fConfig.fNSteps > 0);

  // Step-size array must have the same length as the parameter array
  assert(fConfig.fStepSize.size() == fModel->get_parameter_values().size());

  // Can't have diagonal AND dense
  assert(!(fConfig.matrixDense && fConfig.matrixDiagonal));

  // If used, the matrix size must correspond to the number of parameters
  if(fConfig.matrixDense || fConfig.matrixDiagonal){

    assert(static_cast<int>(fConfig.covariance.rows()) == static_cast<int>(fModel->get_parameter_values().size()));

    assert(static_cast<int>(fConfig.covariance.cols()) == static_cast<int>(fModel->get_parameter_values().size()));
  }
}

// Generates the output file and initializes the TTree
//template <class Model>
void MetropolisHastings::initialize_sampler()
{
  std::cout << "Initialising the Metropolis-Hasting sampler" << std::endl;
  // Make sure whatever is in the config makes sense
  validate_config();

  fConfig.seed = 999;
  // Initialize the random number generator
  random = new TRandom3(fConfig.seed);

  // Initialize the parameters. Model should return the parameter values that
  // we want to start MCMC from -- this could be the nominal, or random around
  // the nominal.
  fCurrent.varValues= fModel->get_parameter_values();
  fCurrent.varNames = fModel->get_parameter_names();
  fCurrent.logProb = fModel->log_prob(fCurrent.varValues);
  fProposed = fCurrent;

  // Number of fitting parameters;
  fNPars = fCurrent.varValues.size();

  // Get the step-sizes. 
  fStepSizes = fConfig.fStepSize;

  nAccepted = 0;

  // Cholesky-decompose the covariance matrix
  if(fConfig.matrixDense){
    std::cout << "Using a dense matrix!" << std::endl;
    Eigen::LLT<Eigen::MatrixXd> tmp(fConfig.covariance);
    choleskyLLT = tmp.matrixL();
  }

  // Create the output file & tree
  fOutFile = new TFile(fConfig.fOutputFile.c_str(), "RECREATE");
  fOutFile->cd();
  fOutTree = new TTree("samples", "MCMC samples");
  fOutTree->SetAutoSave(-10E6);
  std::cout << "Output will be saved in " << fConfig.fOutputFile << std::endl;

  // Create branch per variable
  for(int i = 0; i < fNPars; ++i)
    fOutTree->Branch(fCurrent.varNames[i].c_str(), &fCurrent.varValues[i]);

  fOutTree->Branch("log_prob", &fCurrent.logProb);
  fOutTree->Branch("step", &fStep);
  std::cout << "Output initialized" << std::endl;
}

// Runs the MCMC sampler!
//template <class Model>
void MetropolisHastings::run_mcmc()
{
  for (fStep = 0; fStep < fConfig.fNSteps; ++fStep) {

    // Propose a new MCMC step
    propose_step();

    // Do the Metropolis-Hastings step comparison
    metropolis_correction();

    // Fill the ttree with current steps
    fOutTree->Fill();
  }

  std::cout << "Accepted: " << nAccepted<<"/"<<fConfig.fNSteps << " :: " << (float(nAccepted)/float(fConfig.fNSteps))*100.0 << "%" << std::endl;

  fOutTree->Write();
}

// New MCMC step. Update the step-sizes before updating the parameter values
// TODO: I keep multiplying the exact same fConfig.fStepSizes and
//       covariance(i,i) for the diagonal option, should change that.
//template <class Model>
void MetropolisHastings::propose_step()
{
  // Randomize the step-sizes
  for(int i = 0; i < fNPars; ++i)
    fStepSizes[i] = random->Gaus(0,1);

  // Correlate random values if we use dense matrix. The way we mapped the
  // vector into Eigen means we are actually changing the fStepSizes.
  // Otherwise, if we use the diagonal matrix, multiply random values by the
  // diagonal
  if(fConfig.matrixDense){
    //Eigen::Map<Eigen::VectorXd> tmpeigen = fStepSizes.data();
    Eigen::VectorXd tmpeigen = Eigen::Map<Eigen::VectorXd>(fStepSizes.data(), fStepSizes.size());
    tmpeigen = choleskyLLT*tmpeigen;

    for(int i = 0; i< fNPars; ++i)
      fStepSizes[i] = tmpeigen(i);
  }
  else if(fConfig.matrixDiagonal){
    for(int i = 0; i < fNPars; ++i)
      fStepSizes[i] *= fConfig.covariance(i,i);
  }

  // Scale the step-sizes by user-defined values
  for (int i = 0; i < fNPars; ++i)
    fStepSizes[i] *= fConfig.fStepSize[i];

  // Finally, update the parameter values
  for (int i = 0; i < fNPars; ++i)
    fProposed.varValues[i] = fCurrent.varValues[i] + fStepSizes[i];

  // Update the log-probability 
  fProposed.logProb = fModel->log_prob(fProposed.varValues);
}

// Metropolis-Hastings correction
//template <class Model>
void MetropolisHastings::metropolis_correction()
{
  // Acceptance ratio from the probability ratio
  double acceptRatio = std::exp(-fProposed.logProb + fCurrent.logProb);

  // Random Metropolis-Hastings throw
  double metHastRnd = random->Rndm();

  // If accepted, overwrite the current with proposed values
  if(metHastRnd <= acceptRatio){
    fCurrent = fProposed;
    nAccepted++;
  }
}


//} /* mcmc */ 
