#include "model_covariance.hpp"

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
  // Get the inverse matrix now, so we don't have to make it every logl
  // evaluation
  covarianceInv = covariance.inverse();

  // Sanity check
  assert(covariance.cols() == covariance.rows());

  // Save the number of parameters
  nPars = covariance.rows();

  // Initialise the default parameter names, nominals, and current values. This
  // can all be changed with setters.
  for(int i = 0; i < nPars; ++i)
  {
    names.push_back(("covarianceModel_" + std::to_string(i)).c_str());
    nominal.push_back(0.0);
    values.push_back(0.0);
  }
}

// Set custom parameter values
void ModelCovariance::set_parameter_values(std::vector<double> _values)
{
  assert(static_cast<int>(_values.size()) == nPars);
  values = _values;
}

// Set custom nominal values
void ModelCovariance::set_nominal_values(std::vector<double> _nominal)
{
  assert(static_cast<int>(_nominal.size()) == nPars);
  nominal = _nominal;
}

// Set custom parameter names
void ModelCovariance::set_parameter_names(std::vector<std::string> _names)
{
  assert(static_cast<int>(_names.size()) == nPars);
  names = _names;
}

// Get the log probability. This is just a simple multivariate gaussian, so not
// much to do here.
double ModelCovariance::log_prob(std::vector<double> pars)
{
  // We are assuming the supplied parameter order is the same as the internal.
  assert(static_cast<int>(pars.size()) == nPars);
  double log_prob = 0;
  for(int i = 0; i < nPars; ++i){
    for(int j = 0; j < nPars; ++j){
      log_prob += 0.5*(pars[i] - nominal[i])*(pars[j] - nominal[j])*covarianceInv(i,j);
    }
  }
  return log_prob;
}
