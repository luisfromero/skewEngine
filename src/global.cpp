#include "global.h"

std::chrono::time_point<std::chrono::high_resolution_clock> startCPU, startCPU2, startGPU, endCPU, endCPU2, endGPU;
std::chrono::duration<double, std::nano> elapsed = std::chrono::nanoseconds::zero();



std::vector<short> pixels;
unsigned imgWidth,imgHeight;



bool verbose=true;
float *inputD, *outD;
inputData<float> inData;

/*
bool compactar = false;
bool borrar = true;
bool modotiming = true;
bool mododibujo = false;
bool modofloat = true;
bool silent=false;
bool inputmask=false,fullstore=false;
int deviceIndex=0;

float **resultado;
double **profileX;
double **profileY;
int *desti;     // destino
float *rat;     // ratio
int mode = 0;
int runmode = -1; //not set

*/

//skewEngine<float>  *skewer;
//float inversos[10000];




//**********************************************************************************
//                   Sample points  2023
//**********************************************************************************

int nSamples=3;
bool saveSampleData=false;
std::vector<point_t>  samplePoints;
std::vector<std::vector<float>>  sampleData0[180];
std::vector<std::vector<float>>  sampleData1[180];
std::vector<std::vector<float>>  sampleData2[180];
std::vector<std::vector<float>>  sampleData3[180];
point_t punto1={1082,562};
point_t punto2={348,266};
point_t punto3={600,536};


//**********************************************************************************
//                   SKEWENGINE executable  2023  (globals)
//**********************************************************************************

int runMode=GPU_MODE; // 0 cpu, 1 gpu, 2 hybrid task farm en el futuro
int gpuMode=OPENCL_MODE; // 0 cpu, 1 gpu, 2 hybrid task farm en el futuro

int nthreads = -1;//not set
int nCPUs,nGPUs;
int dim,dimx,dimy,N;  //N and dim are aliases
int maxSector=180;

color *my_palette;

