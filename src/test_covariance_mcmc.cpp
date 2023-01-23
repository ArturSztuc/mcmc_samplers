#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "model/model.hpp"
#include "model/model_covariance.hpp"
#include "mcmc/mcmc_methastings.hpp"

#include "diagnostics/diag_autoc.hpp"
#include "diagnostics/diag_traces.hpp"
#include "diagnostics/diag_posterior.hpp"
#include "utils/progressbar.hpp"
#include "/home/artur/rootlogon_fixedl.C"
#include "/home/artur/work/nova/scripts/utils/utils.C"

#include "TMatrixDSym.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TAxis.h"
#include "TH1D.h"
#include "TPad.h"
#include "TList.h"
#include "TStopwatch.h"

#include "THashList.h"

int main(int argc, char *argv[])
{
  rootlogon_fixedl();

  // Use one of the T2K covariance matrices for tests
  TFile file_cov("/home/artur/work/t2k/MaCh3/inputs/BANFF_DataFit_2018_final_180712_sk.root");
  TMatrixDSym *cov = (TMatrixDSym*)file_cov.Get("postfit_cov");

  // Initialise the model
  ModelBase *mod = new ModelCovariance(cov);
  std::vector<std::string> names = mod->get_parameter_names();

  // Initialise the step-sizes.
  std::vector<double> step_sizes;
  int selected_id = 0;
  for (int i = 0; i < cov->GetNrows(); ++i) {
    // Additionally optimised with the dense matrix
    step_sizes.push_back(0.14); 
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
  config.seed = rand() % 10000;
  config.fNSteps = 1000000;
  config.fStepSize = step_sizes;
  config.fOutputFile = "out.root";
  // Cholesky decomp
  config.matrixDense = true;
  // Diagonal matrix for tuning
  config.matrixDiagonal = false;
  // Not needed if the two options above are false
  config.covariance = covariance;

  TStopwatch clock;

  MetropolisHastings sampler(mod, config);
  clock.Start();
  sampler.run_mcmc();
  clock.Stop();

  double tme = clock.RealTime();
  std::cout << "MCMC done in: " << tme << "s , or: " << tme/config.fNSteps << "s/step\n";

  // Do the diagnostics
  std::cout << "Diagnostics: " << std::endl;
  DiagTraces *traceGen = new DiagTraces(sampler.get_tree());

  TCanvas c(" ", " ", 800, 600);
  TGraph *gr = traceGen->GetTrace(0);
  gr->GetXaxis()->SetRangeUser(0, config.fNSteps);
  gr->SetTitle(";MCMC Step index;Parameter Value");
  gr->Draw("AP");
  CenterTitles(gr);
  c.SaveAs("test.png");

  // Do the 1D
  DiagPosterior *posteriorGen = reinterpret_cast<DiagPosterior*>(traceGen);
  for (int i = 0; i < cov->GetNrows(); ++i) {
    c.SetLeftMargin(0.15);
    c.SetBottomMargin(0.15);
    TH1D *h = posteriorGen->GetPosteriorProbability(i);
    h->Scale(1./h->Integral());
    std::string s = ";";
    s += "Parameter Value"; //names[i];
    s += ";Posterior Probability Density";
    h->SetTitle(s.c_str());

    TH1D* h68 = GetInterval(h, 0.6827);
    TH1D* h95 = GetInterval(h, 0.9545);
    TH1D* h99 = GetInterval(h, 0.9973);

    SetColours1D_NO(h, h68, h95, h99);

    h->Draw("hist");
    h68->Draw("hist same");
    h95->Draw("hist same");
    h99->Draw("hist same");
    std::string ss = "posteriorProbability_1D_";
    c.SaveAs((ss + names[i] + ".png").c_str());
  }

  DiagAutocorrelations *autoGen = reinterpret_cast<DiagAutocorrelations*>(traceGen);
  progressbar bar(cov->GetNrows());
  bar.set_opening_bracket_char("Autocorrelations: [");
  TGraph *selected;
  for (int i = 0; i < cov->GetNrows(); ++i) {
    bar.update();
    TGraph *g = autoGen->GetAutocorrelation(i, 10000);

    if((i == selected_id) || (i == 2)){
      g->SetLineColor(kRed);
      g->SetMarkerColor(kRed);
      selected = g;
    }
    if(i == 0){
      selected = g;
      g->GetYaxis()->SetRangeUser(-0.3,1);
      g->Draw("ap");
    }
    else
      g->Draw("p same");
  }
  selected->Draw("p same");
  std::cout << std::endl;
  c.SaveAs("autocors.png");



  return 0;
}
