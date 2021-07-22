#pragma once
#ifndef MODELBASE_H
#define MODELBASE_H

#include <vector>
#include <string>

// Base model class that all the other models are derived from. All models
// absolutely need to have log_prob, and get_parameter_values.
// TODO:
//  * Think about how to treat gradients here. All models should have it
//    implemented, at the moment only few models can be used with
//    Hamiltonian-derived MCMCs

class ModelBase
{
public:
  ModelBase (){};
  virtual ~ModelBase (){};

  virtual double log_prob(std::vector<double> pars) = 0;

  virtual void set_parameter_values(std::vector<double> pars) {}
  virtual void set_parameter_names(std::vector<std::string> names) {}

  virtual std::vector<double> get_parameter_values() = 0;
  virtual std::vector<std::string> get_parameter_names() = 0;
};
#endif /* MODELBASE_H */
