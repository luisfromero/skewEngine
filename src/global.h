//#pragma once
#ifndef SKE_GLOBAL_H
#define SKE_GLOBAL_H
#include <string>
#include <chrono>
#include <vector>
#include "skewEngine.cuh"
#include "defaults.h"
#include "structures.h"
#ifdef _OPENMP
#define OPENMP
#endif





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
extern std::vector<std::vector<float>>  sampleData1[180];
extern std::vector<std::vector<float>>  sampleData2[180];
extern std::vector<std::vector<float>>  sampleData3[180];
extern point_t punto1;
extern point_t punto2;
extern point_t punto3;


//**********************************************************************************
//                   SKEWENGINE  2023  (generic)
//**********************************************************************************

extern int runMode; // 0 cpu, 1 gpu, 2 hybrid task farm en el futuro
extern int nthreads ;
extern int nCPUs,nGPUs;
extern int dim,dimx,dimy,N;
extern int maxSector;
extern bool verbose;



extern unsigned imgWidth,imgHeight;
extern std::vector<short> pixels;



#endif