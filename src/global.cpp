#include "global.h"

std::chrono::time_point<std::chrono::high_resolution_clock> startCPU, startCPU2, startGPU, endCPU, endCPU2, endGPU;
std::chrono::duration<double, std::nano> elapsed = std::chrono::nanoseconds::zero();
bool compactar = false;
bool borrar = true;
bool modotiming = true;
bool mododibujo = false;
bool modofloat = true;
bool verbose=true;
bool silent=false;



int utm_n,utm_e;
int deviceIndex=0;
bool floatsDEM = false, endianOK = true;
bool inputmask=false,fullstore=false;


double *heights;
float *h_DEM; //replace for heights

float *inputD, *outD;
inputData<float> inData;


float **resultado;
double **profileX;
double **profileY;
int *desti;     // destino
float *rat;     // ratio






int mode = 0;
int runmode = -1; //not set

GpuInterface *gpu;


//skewEngine<float>  *skewer;
//float inversos[10000];



//**********************************************************************************
//                   SDEM  2023
//**********************************************************************************

double obsheight = OBS_H;
float POVh = OBS_H;
float step=STEP;
float surScale =M_PI/(360*STEP*STEP); //Always assuming precision 1 degree
float volScale =M_PI/(3*360*STEP*STEP*STEP); //Volumetric viewshed


//**********************************************************************************
//                   BLUR  2023
//**********************************************************************************

std::vector<short> pixels;
unsigned imgWidth,imgHeight;

//**********************************************************************************
//                   SKEWENGINE  2023  (generic)
//**********************************************************************************

int runMode=0; // 0 cpu, 1 gpu, 2 hybrid task farm en el futuro
color *my_palette;
CpuInterfaceV3 *cpu;
GpuInterfaceV3 *gpuV3;
int nthreads = -1;//not set
int nCPUs,nGPUs;
int dim,dimx,dimy,N;  //N and dim are aliases
int maxSector=180;

