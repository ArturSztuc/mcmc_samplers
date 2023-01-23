#include "mcmc_methastings.hpp"

MetropolisHastings::MetropolisHastings(ModelBase *_fModel, Config _fConfig)
  : MCMCBase(), fModel(_fModel), fConfig(_fConfig)
{
  // Initialize the sampler
  initialize_sampler();
  Eigen::initParallel();
  omp_set_num_threads(16);
  Eigen::setNbThreads(16);
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

  generator = new std::default_random_engine(fConfig.seed);

  // Initialize the parameters. Model should return the parameter values that
  // we want to start MCMC from -- this could be the nominal, or random around
  // the nominal.
  std::vector<double> tmpVec = fModel->get_parameter_values();
  fNPars = tmpVec.size();
  fCurrent.vals = Eigen::VectorXd::Ones(tmpVec.size());
  for(size_t i = 0; i < tmpVec.size(); ++i)
    fCurrent.vals(i) = tmpVec[i];

  fCurrent.varNames   = fModel->get_parameter_names();
  fCurrent.logProb    = fModel->log_prob(fCurrent.vals);
  fProposed = fCurrent;

  // Get the step-sizes. 
  fStepSizes = Eigen::Map<Eigen::VectorXd>(fConfig.fStepSize.data(), fConfig.fStepSize.size());

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
    fOutTree->Branch(fCurrent.varNames[i].c_str(), &fCurrent.vals(i));

  fOutTree->Branch("log_prob", &fCurrent.logProb);
  fOutTree->Branch("step", &fStep);
  std::cout << "Output initialized" << std::endl;
}

// Runs the MCMC sampler!
void MetropolisHastings::run_mcmc()
{

  progressbar bar(fConfig.fNSteps);
  bar.set_opening_bracket_char("Running MCMC: [");
  for (fStep = 0; fStep < fConfig.fNSteps; ++fStep) {
    bar.update();

    // Propose a new MCMC step
    propose_step();

    // Do the Metropolis-Hastings step comparison
    metropolis_correction();

    // Fill the ttree with current steps
    fOutTree->Fill();
  }
  std::cout << std::endl;

  std::cout << "Accepted: " << nAccepted<<"/"<<fConfig.fNSteps << " :: " << (float(nAccepted)/float(fConfig.fNSteps))*100.0 << "%" << std::endl;

  fOutTree->Write();
}

// New MCMC step. Update the step-sizes before updating the parameter values
// TODO: I keep multiplying the exact same fConfig.fStepSizes and
//       covariance(i,i) for the diagonal option, should change that.
void MetropolisHastings::propose_step()
{
  // Randomize the step-sizes
  for(int i = 0; i < fNPars; ++i){
    std::normal_distribution<double> distribution(0.0, fConfig.fStepSize[i]);
    fStepSizes(i) = distribution(*generator);
  }

  // Correlate random values if we use dense matrix. The way we mapped the
  // vector into Eigen means we are actually changing the fStepSizes.
  // Otherwise, if we use the diagonal matrix, multiply random values by the
  // diagonal
  if(fConfig.matrixDense)
    fStepSizes = choleskyLLT*fStepSizes;
  else if(fConfig.matrixDiagonal){
    for(int i = 0; i < fNPars; ++i)
      fStepSizes(i) *= fConfig.covariance(i,i);
  }

  // Finally, update the parameter values
  fProposed.vals = fCurrent.vals + fStepSizes;

  //// Update the log-probability 
  fProposed.logProb = fModel->log_prob(fProposed.vals);
}

// Metropolis-Hastings correction
//template <class Model>
void MetropolisHastings::metropolis_correction()
{
  // Acceptance ratio from the probability ratio
  double acceptRatio = std::exp(-fProposed.logProb + fCurrent.logProb);

  // Random Metropolis-Hastings throw
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  double metHastRnd = distribution(*generator);

  //double metHastRnd = random->Rndm();

  // If accepted, overwrite the current with proposed values
  if(metHastRnd <= acceptRatio){
    fCurrent = fProposed;
    nAccepted++;
  }
}
