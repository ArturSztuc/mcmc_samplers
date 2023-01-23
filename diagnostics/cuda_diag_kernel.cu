// std
#include <stdio.h>

// CUDA specifics
#include <cuda_runtime.h>
//#include <helper_functions.h>
//#include <helper_cuda.h>

// Debugging macros
#define CUDA_ERROR_CHECK
#define CudaCheckError()  __cudaCheckError(__FILE__, __LINE__)


// Check if there's been an error
inline void __cudaCheckError( const char *file, const int line ) {
  // **************************************************
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }
#endif
  return;
}

__global__ void CalculateAutocorrelations(
    float * gParameterValues,
    float * gMeansGPU,
    int gParSizeAll,
    int gParSize,
    int gMaxLag,
    float * gAutocorrelations
    )
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int bdim  = blockDim.x * gridDim.x;

  // Length of autocorrelation array
  int gParLag = gParSize * gMaxLag;
  // Number of MCMC steps
  int nSteps  = gParSizeAll/gParSize;

  // Is there a way to make this a matrix multiplication using Eigen?
  // parameter x step
  // parameter x lag

  for(int thread = index; thread < gParLag; thread += bdim)
  {
    int iPar = (thread / gMaxLag) % gParSize;
    int iLag = thread % gMaxLag;

    float numerator = 0;
    float denominator = 0;
    for(int i = 0; i < nSteps; ++i){
      int idxtmp = (iPar * nSteps) + i;
      float diff = gParameterValues[idxtmp] - gMeansGPU[iPar];

      if(i < nSteps - iLag){
        float lagTerm = gParameterValues[idxtmp + iLag] - gMeansGPU[iPar];
        float product = diff*lagTerm;
        numerator += product;
      }
      denominator += diff * diff;
    }
    gAutocorrelations[(iPar * gMaxLag) + iLag] = numerator/denominator;
  }
}

// Allocate memory on GPU
__host__ void InitValuesGPU(
    float **gParameterValues,
    float **gAutocorrelations,
    float **gMeans,
    int gParSizeAll, 
    int gMaxLag,
    int gNPars)
{
  cudaMalloc((void**) gParameterValues, gParSizeAll*sizeof(float));
  CudaCheckError();
  printf("Allocated %i parameter entries with total memory of %f Mb\n", gParSizeAll, (gParSizeAll*sizeof(float))/1.E6);

  cudaMalloc((void**) gAutocorrelations, gMaxLag*gNPars*sizeof(float));
  CudaCheckError();
  printf("Allocated %i autocorrelation entries with total memory of %f Mb\n", gMaxLag*gNPars, (gMaxLag*gNPars*sizeof(float))/1.E6);

  cudaMalloc((void**) gMeans, gNPars*sizeof(float));
  CudaCheckError();
  printf("Allocated %i means with total memory of %f Mb\n", gNPars, (gNPars*sizeof(float))/1.E6);
}

// Copies CPU RAM to GPU
__host__ void CopyToGPU(
    float *gParameterValues,
    float *fParameterValues,
    float *gMeans,
    float *fMeans,
    int fParSizeAll,
    int fParSize
    )
 
{
  cudaMemcpy(gParameterValues, fParameterValues, fParSizeAll*sizeof(float), cudaMemcpyHostToDevice);
  CudaCheckError();
  printf("Copied parameter values to the GPU\n");

  cudaMemcpy(gMeans, fMeans, fParSize*sizeof(float), cudaMemcpyHostToDevice);
  CudaCheckError();
  printf("Copied parameter means to the GPU\n");
}

__host__ void RunAutocorrelationsGPU(
      float *gParameterValues,
      float *gAutocorrelations,
      float *gMeansGPU,
      int gParSizeAll, 
      int gParSize,
      int gMaxLag,
      float *fAutocorrelations)
{

  int block_size = 1024;
  int num_blocks = ((gMaxLag * gParSize) + block_size - 1) / block_size;

  printf("Calculating Autocorrelations...\n");
  CalculateAutocorrelations<<<num_blocks, block_size>>>(
      gParameterValues,
      gMeansGPU,
      gParSizeAll,
      gParSize,
      gMaxLag,
      gAutocorrelations
      );
  CudaCheckError();

  printf("Copying Autocorrelations from the GPU back to Host RAM...\n");
  cudaMemcpy(fAutocorrelations, gAutocorrelations, gParSize * gMaxLag*sizeof(float), cudaMemcpyDeviceToHost);
  CudaCheckError();
}

__host__ void ClearValuesGPU(
    float *gParameterValues,
    float *gAutocorrelations,
    float *gMeansGPU)
{
  cudaFree(gParameterValues);
  cudaFree(gAutocorrelations);
  cudaFree(gMeansGPU);
}
