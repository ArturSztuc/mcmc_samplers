#include "diagnostics.hpp"

Diagnostics::Diagnostics(
    TTree *tree,
    std::string _logprobname,
    int _burnin,
    std::vector<std::string> _inactive
    )
  : fTree(tree), fLogProbName(_logprobname), fBurnIn(_burnin), fInactiveNames(_inactive)
{

  fNPar   = 0;
  fNSteps = 0;
  FillActiveNames();
  FillRAM();

#ifdef CUDA
  std::cout << "CUDA-enabled GPU is available <3\n";
#endif
}

// Fills an array of parameter names to use for diagnostics
void Diagnostics::FillActiveNames()
{
  // Get number of parameters 
  double nBranches = fTree->GetNbranches();

  // Get a list of branches
  TObjArray* branches = (TObjArray*)fTree->GetListOfBranches();

  // Get active branch names
  for (int i = 0; i < nBranches; ++i) {
    std::string tBName = branches->At(i)->GetName();

    // Reject inactive branches
    bool active = true;
    if(tBName == fLogProbName)
      continue;
    for(auto iName : fInactiveNames){
      if(iName == tBName)
        active = false;
    }
    if(!active) continue;

    // Save active names
    fActiveNames.push_back(tBName);
  }
  fNPar = fActiveNames.size();
}

// Fill RAM from TTree
void Diagnostics::FillRAM()
{
  std::cout << "Getting TTree info\n";

  // Throw an error if no active parameters saved
  assert(fNPar > 0);

  // Get the number of MCMC steps
  fNSteps = fTree->GetEntries() - fBurnIn;

  std::cout << "Initializing the storage\n";
  // Initialise the storage
  fParValues.resize(fNPar, std::vector<double>(fNSteps, 0));
  std::vector<double> fLogLValues(fNSteps, 0);

  // Initialise temporary storage to pull from TTree
  std::vector<double> tTmpPar(fNPar, 0);
  double tTmpLogl = 0;

  std::cout << "Setting adresses\n";
  // Set all the branch addresses
  int i = 0;
  for(auto iName : fActiveNames)
    fTree->SetBranchAddress(iName.c_str(), &tTmpPar[i++]);
  fTree->SetBranchAddress(fLogProbName.c_str(), &tTmpLogl);

  std::cout << "Filling RAM <3\n";
  // Now we will iterate over the chain and actually fill RAM
  progressbar bar(fNSteps);
  bar.set_opening_bracket_char("Filling Diagnostic RAM: [");
  for (int i = 0; i < fNSteps; ++i) {
    bar.update();
    fTree->GetEntry(i + fBurnIn);
    for (int j = 0; j < fNPar; ++j) {
      fParValues[j][i] = tTmpPar[j];
    }
    fLogLValues[i] = tTmpLogl;
  }

  delete fTree;
  // Let's not kill RAM
  //delete branches;
}

// Clear all the RAM
void Diagnostics::ClearRAM()
{
  //delete fTree;
  fNPar   = 0;
  fNSteps = 0;

  fLogLValues.clear();
  fParValues.clear();
}

void Diagnostics::PrintTest()
{
  std::cout << "PrintTest: standard matrix/vector layout" << std::endl;
  for(auto v1 : fParValues){
    for(auto v2 : v1){
      std::cout << v2 << std::endl;
    }
  }

  std::cout << "PrintTest: data() layout" << std::endl;
  for(auto v1 :fParValues){
    double *data = v1.data();
    for (int i = 0; i < fNSteps ; ++i) {
      std::cout << data[i] << std::endl;
    }
  }
}
