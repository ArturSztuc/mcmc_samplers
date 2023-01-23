#include "cuda_diagnostics.hpp"

#include "omp.h"

// CUDA functions

// Initializes empty containers on the GPU with parameter values and lags.
extern void InitValuesGPU(
    float **gParameterValues,
    float **gAutocorrelations,
    float **gMeansGPU,
    int gParSizeAll, 
    int gMaxLag,
    int gNPar
    );

// Clears the GPURAM 
extern void ClearValuesGPU(
    float *gParameterValues,
    float *gAutocorrelations,
    float *fMeansCPU
    );

// Copy from CPU to GPU RAM
extern void CopyToGPU(
    float *gParameterValues,
    float *fParameterValues,
    float *gMeans,
    float *fMeans,
    int fParSizeAll,
    int fParSize
    );

//// Run the autocorrelations on GPU and copy final output from GPU to CPU
extern void RunAutocorrelationsGPU(
    float *gParameterValues,
    float *gAutocorrelations,
    float *gMeansGPU,
    int gParSizeAll, 
    int gParSize,
    int gMaxLag,
    float *fAutocorrelations);


CudaDiagnostics::CudaDiagnostics(int _maxlag, const std::vector< std::vector< double > > &_parvals )
  : fMaxLag(_maxlag)
{
  ScanParameters(_parvals);
  fAutocorrelationsCPU = new float[fMaxLag * fParSize];
  fParameterValuesCPU  = new float[fParSizeAll];
  fMeansCPU = new float[fParSize];

  // Prepare the input
  std::cout << "Preparing the input array\n";
  for (int i = 0; i < fParSize; ++i) {
    for (int j = 0;  j< fNSteps; ++j) {
      fParameterValuesCPU[GetIDX(i, j)] = _parvals[i][j];
    }
  }
  std::cout << "Array takes " << (sizeof(float)*fParSizeAll)/1.E6 << "Mb of RAM\n";

  for(int idPar = 0; idPar < fParSize; ++idPar){
    fMeansCPU[idPar] = 0;
    for(int idStep = 0; idStep < fNSteps; ++idStep){
      fMeansCPU[idPar] += fParameterValuesCPU[GetIDX(idPar, idStep)];
    }
    fMeansCPU[idPar] /= fNSteps;
  }

  // Initialize GPU
  std::cout << "Allocating GPU memory\n";
  InitValuesGPU(
      &gParameterValuesGPU,
      &gAutocorrelationsGPU,
      &gMeansGPU,
      fParSizeAll,
      fMaxLag,
      fParSize
      );

  // Copy the input from CPU to GPU
  std::cout << "Copying parameter values from Host RAM to GPU\n";
  CopyToGPU(
      gParameterValuesGPU,
      fParameterValuesCPU,
      gMeansGPU,
      fMeansCPU,
      fParSizeAll,
      fParSize);
}

CudaDiagnostics::~CudaDiagnostics()
{
  ClearValuesGPU(gParameterValuesGPU, gAutocorrelationsGPU, gMeansGPU);
  delete fParameterValuesCPU;
  delete fAutocorrelationsCPU;
  delete fMeansCPU;
}

int CudaDiagnostics::GetIDX(int parameter, int step)
{
  return (parameter * fNSteps) + step;
}

std::vector<TGraph*> CudaDiagnostics::GetAutocorrelations()
{

  RunAutocorrelationsGPU(
      gParameterValuesGPU,
      gAutocorrelationsGPU,
      gMeansGPU,
      fParSizeAll, 
      fParSize,
      fMaxLag,
      fAutocorrelationsCPU);

  std::vector<TGraph*> ret;

  // Create and fill the TGraphs
  for(int iPar = 0; iPar < fParSize; ++iPar){
    TGraph * tmp = new TGraph(fMaxLag);
    for(int iLag = 0; iLag < fMaxLag; ++iLag){
      int idx = (iPar * fMaxLag) + iLag;
      tmp->SetPoint(iLag, iLag, fAutocorrelationsCPU[idx]);
    }
    ret.push_back(tmp);
  }
  return ret;
}

void CudaDiagnostics::ScanParameters(const std::vector< std::vector< double > > &_parvals)
{
  fParSize  = _parvals.size();
  fNSteps   = _parvals[0].size();
  fParSizeAll = fParSize * fNSteps;
  printf("Parameters: %i, Steps: %i, Total: %i", fParSize, fNSteps, fParSizeAll);
}
