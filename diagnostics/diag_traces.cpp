#include "diag_traces.hpp"

DiagTraces::DiagTraces(TTree *tree) : Diagnostics(tree)
{
}

// Get a TGraph
TGraph* DiagTraces::GetTrace(int parameter)
{
  double min =  *std::min_element(fParValues[parameter].begin(), fParValues[parameter].end());
  double max =  *std::max_element(fParValues[parameter].begin(), fParValues[parameter].end());
  TGraph* ret = new TGraph(fNSteps);
  for (int i = 0; i < fNSteps; ++i) {
    ret->SetPoint(i, i, fParValues[parameter][i]);
  }

  ret->GetYaxis()->SetRangeUser(min, max);
  return ret;
}
