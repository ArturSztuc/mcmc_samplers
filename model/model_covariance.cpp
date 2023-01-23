#include "model_covariance.hpp"
#include <limits>

// Initialise the covariance matrix model using the Eigen's matrix object
ModelCovariance::ModelCovariance(Eigen::MatrixXd _covariance)
  : ModelBase() 
{
  covariance = _covariance;
  initialize_model();
}

// Initialise the covariance matrix model using Root's TMatrixDSym object
ModelCovariance::ModelCovariance(TMatrixDSym* _covariance)
  : ModelBase()
{
  std::cout << "Setting the ModelCovariance model" << std::endl;

  // There must be better ways of doing this...
  double *tmp = _covariance->GetMatrixArray();
  covariance = Eigen::Map<Eigen::MatrixXd>(tmp, 
                                    _covariance->GetNrows(),
                                    _covariance->GetNcols());

  initialize_model();
}

// Initialise the covariance model.
void ModelCovariance::initialize_model()
{
  omp_set_num_threads(16);
  Eigen::setNbThreads(16);

  // Get the inverse matrix now, so we don't have to make it every logl
  // evaluation
  covarianceInv = covariance.inverse();

  // Sanity check
  assert(covariance.cols() == covariance.rows());

  // Save the number of parameters
  nPars = covariance.rows();

  fNotFlatPriors = Eigen::VectorXd::Ones(nPars);

  // Initialise the default parameter names, nominals, and current values. This
  // can all be changed with setters.
  nominal = Eigen::VectorXd::Zero(nPars);
  values  = Eigen::VectorXd::Zero(nPars);

  for(int i = 0; i < nPars; ++i)
    names.push_back(("covarianceModel_" + std::to_string(i)).c_str());
}

void ModelCovariance::set_flat_prior(int idx)
{ 
  fNotFlatPriors(idx) = 0; 
}

// Set custom parameter values
void ModelCovariance::set_parameter_values(std::vector<double> _values)
{
  assert(static_cast<int>(_values.size()) == nPars);
  for (int i = 0; i < nPars; ++i)
    values(i) = _values[i];
}

// Set custom nominal values
void ModelCovariance::set_nominal_values(std::vector<double> _nominal)
{
  assert(static_cast<int>(_nominal.size()) == nPars);
  for (int i = 0; i < nPars; ++i)
    nominal(i) = _nominal[i];
}

// Set custom parameter names
void ModelCovariance::set_parameter_names(std::vector<std::string> _names)
{
  assert(static_cast<int>(_names.size()) == nPars);
  names = _names;
}

// Get the log probability. This is just a simple multivariate gaussian, so not
// much to do here.
double ModelCovariance::log_prob(const Eigen::VectorXd &pars)
{
  double log_prob = 0;

  // We are assuming the supplied parameter order is the same as the internal.
  assert(static_cast<int>(pars.size()) == nPars);

  Eigen::VectorXd diff = pars - nominal;

  diff = diff.array() * fNotFlatPriors.array();

  log_prob += diff.matrix().dot(covarianceInv*diff.matrix());

  return log_prob;
}
