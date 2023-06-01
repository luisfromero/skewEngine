//#pragma once
#ifndef SKE_GLOBAL_H
#define SKE_GLOBAL_H


#ifdef _OPENMP
#define OPENMP
#endif




#include <string>
#include <chrono>
#include <vector>
#include "skewEngine.cuh"
#include "defaults.h"
#include "structures.h"





extern std::chrono::time_point<std::chrono::high_resolution_clock> startCPU, startCPU2, startGPU, endCPU, endCPU2, endGPU;
extern std::chrono::duration<double, std::nano> elapsed;

extern float *inputD;
extern float *outD;
extern inputData<float> inData;

/*
extern float **resultado;
extern double **profileX;
extern double **profileY;
extern int *desti;     // destino
extern float *rat;     // ratio


extern int mode ;
extern int runmode;
*/

//Habrá que ir borrando


//**********************************************************************************
//                   Samples  2023
//**********************************************************************************


extern int nSamples;
extern bool saveSampleData;
extern std::vector<point_t>  samplePoints;
extern std::vector<std::vector<float>>  sampleData0[180];
extern std::vector<std::vector<float>>  sampleData1[180];
extern std::vector<std::vector<float>>  sampleData2[180];
extern std::vector<std::vector<float>>  sampleData3[180];
extern point_t punto1;
extern point_t punto2;
extern point_t punto3;


//**********************************************************************************
//                   SKEWENGINE  2023  (algunas consideraciones generales)
//**********************************************************************************

enum RUN_MODE{CPU_MODE,GPU_MODE,HYBRID_MODE};
enum GPU_MODE{CUDA_MODE,OPENCL_MODE};
enum KERNELS{KERNEL_SDEM, KERNEL_IDENTITYDEM, KERNEL_RADON,KERNEL_IDENTITYIMG,KERNEL_CEPSTRUM};

extern int runMode; // 0 cpu, 1 gpu, 2 hybrid task farm en el futuro
extern int nthreads ;
extern int nCPUs,nGPUs;
extern int gpuMode;

extern int dim,dimx,dimy,N;
extern int maxSector;
extern bool verbose;



extern unsigned imgWidth,imgHeight;
extern std::vector<short> pixels;



#endif