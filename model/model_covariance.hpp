#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>

#include "TRandom3.h"
#include "TMatrixDSym.h"

#include "model/model.hpp"

#include "omp.h"

#ifndef MODEL_COVARIANCE_H
#define MODEL_COVARIANCE_H

//namespace model
//{

// ModelCovariance model. It samples log-probability from a covariance matrix,
// so it's one way of implementing correlated priors (or, alternatively, can be
// used for simple mcmc diagnostics/tests). Requires an input matrix, either
// ROOT TMatrixDSym or Eigen TMatrixXd
//
// TODO:
//  * Get rid of root's random number generator and replace with standard C++ one
//  * Implement gradient calc for hamiltonian-derived mcmc samplers, sick of
//    using skdetassymetry class for this. 
//  * Need to think how to join covariance + data gradients in model_multimodel.
//  * The way I use this in ensemble samplers (thousands of class copies...) is
//    stupid. Need some internal way of using with multiple class states? +
//    parallelization?
//  * Need to think of some parameter class that can be used across all the
//    different objects... it would make debugging easier/bugs more difficult.
class ModelCovariance : public ModelBase
{
// Class members
private:
  // Nominal parameter values
  Eigen::VectorXd nominal;

  // Parameter values
  Eigen::VectorXd values;

  Eigen::VectorXd fNotFlatPriors;


  // Parameter names
  std::vector<std::string> names;

  // Covariance matrix holder
  Eigen::MatrixXd covariance;
  Eigen::MatrixXd covarianceInv;

  std::vector< std::vector<double> > mat;

  // Number of parameters
  int nPars;

  // Random number generator
  TRandom3 *rnd;

// Public class member functions
public:
  ModelCovariance(Eigen::MatrixXd _covariance);
  ModelCovariance(TMatrixDSym *_covariance);

  // Returns the log-probability given the input parameter values
  double log_prob(const Eigen::VectorXd &pars);

  //double log_prob(std::vector<double> pars, double temp);

  void set_flat_prior(int idx);


  // Returns gradient for parameter idx
  double grad(int idx, double par);

  // Returns gradient vector
  std::vector<double> grad(std::vector<double> pars);

  // Set the parameter values
  void set_parameter_values(std::vector<double> pars);
  // Set the nominal values
  void set_nominal_values(std::vector<double> pars);
  // Set the parameter names
  void set_parameter_names(std::vector<std::string> names);

  // Get the parameter values
  std::vector<double> get_parameter_values(){
    std::vector<double> ret;
    for (int i = 0; i < nPars; ++i) {
      ret.push_back(values(i));
    }
    return ret;
  };

  // Get the parameter names
  std::vector<std::string> get_parameter_names(){return names;};

// Private class member functions
private:
  void initialize_model();
};

#endif /* MODEL_COVARIANCE_H */
