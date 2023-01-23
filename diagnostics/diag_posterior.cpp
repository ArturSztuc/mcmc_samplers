#include "diag_posterior.hpp"



DiagPosterior::DiagPosterior(TTree *tree) : Diagnostics(tree)
{
}

TH1D* DiagPosterior::GetPosteriorProbability(
    int parameter,
    PosteriorConfig cfg)
{
  TH1D* posterior;

  double min, max;
  if(cfg.minx == cfg.maxx){
    min =  *std::min_element(fParValues[parameter].begin(), fParValues[parameter].end());
    max =  *std::max_element(fParValues[parameter].begin(), fParValues[parameter].end());
  }else{
    min = cfg.minx;
    max = cfg.maxx;
  }

  posterior = new TH1D(" ", " ", cfg.nbinsx, min, max);
  for (int i = cfg.burnin; i < fNSteps; ++i) {
    posterior->Fill(fParValues[parameter][i]);
  }

  if(cfg.smooth)
    posterior->Smooth();

  if(cfg.normalize)
    posterior->Scale(1./posterior->Integral());

  return posterior;
}

TH2D* DiagPosterior::GetPosteriorProbability(int parameter1, int parameter2)
{
  TH2D* posterior;
  double min1 =  *std::min_element(fParValues[parameter1].begin(), fParValues[parameter1].end());
  double max1 =  *std::max_element(fParValues[parameter1].begin(), fParValues[parameter1].end());
  double min2 =  *std::min_element(fParValues[parameter2].begin(), fParValues[parameter2].end());
  double max2 =  *std::max_element(fParValues[parameter2].begin(), fParValues[parameter2].end());

  posterior = new TH2D(" ", " ", 100, min1, max1, 100, min2, max2);
  for (int i = 0; i < fNSteps; ++i) {
    posterior->Fill(fParValues[parameter1][i], fParValues[parameter1][i]);
  }



  return posterior;
}

TH1D* DiagPosterior::GetInterval(TH1D* posterior, double interval)
{
  std::string retname = "th_";
  TH1D *copyth  = (TH1D*)posterior->Clone("copyth");
  TH1D* thint   = (TH1D*)posterior->Clone((retname + std::to_string(interval)).c_str());

  double integral = posterior->Integral();
  double tsum = 0;
  while((tsum / integral) <= interval)
  {
    double tmax = copyth->GetMaximum();
    tsum += tmax;
    int bin = copyth->GetMaximumBin();

    copyth->SetBinContent(bin, -1);
    thint->SetBinContent(bin, 0);
  }

  delete copyth;
  return thint;
}
