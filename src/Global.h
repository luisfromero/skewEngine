//#pragma once
#ifndef SKE_GLOBAL_H
#define SKE_GLOBAL_H
#include <string>
#include <chrono>
#include "skewEngine.cuh"
#include "GpuSolver.cuh"
#include "defaults.h"
#ifdef _OPENMP
#define OPENMP
#endif

#ifndef MYSTRUCTS
#define MYSTRUCTS

struct color {unsigned char R,G,B;};
struct hsvColor {float H,S,V;};
struct rgbColor {unsigned char R,G,B;};

typedef struct  {
    float min;
    float max;
}    pair_t;

typedef struct  {
    int x;
    int y;
}    point_t;


typedef struct  {
    int dimx;
    int dimy;
    int bw;
    double obs_h;
    double step;
} sectorgeometry_t ;


typedef struct {
    int nitems;
    char * keys[50];
    char * values[50];
}header_t;





#endif

#include "color.h"
#include "auxf.cuh"


// typedef std::chrono::high_resolution_clock Clock;

// extern int number, numberpop;
// extern double ds;
extern std::chrono::time_point<std::chrono::high_resolution_clock> startCPU, startCPU2, startGPU, endCPU, endCPU2, endGPU;
extern std::chrono::duration<double, std::nano> elapsed;

extern bool compactar ;
extern bool borrar;
extern bool modotiming;
extern bool mododibujo;
extern bool modofloat;
extern bool verbose;
extern bool silent;


extern double *heights;
extern float *h_DEM; //Input 0, DEM

// General use of skewEngine. It will be a template
extern float *inputD;
extern float *outD;
extern inputData<float> inData;

extern float **resultado;
extern double **profileX;
extern double **profileY;
extern int *desti;     // destino
extern float *rat;     // ratio


extern int mode ;
extern int runmode;


//extern skewEngine<float>  *skewer;
extern GpuInterface *gpu;

//extern float inversos[10000]; //experimento


//Habrá que ir borrando

extern int utm_n,utm_e;
extern int deviceIndex;
extern bool floatsDEM, endianOK ;
extern bool inputmask,fullstore;
extern int maxSector;

//**********************************************************************************
//                   SDEM  2023
//**********************************************************************************

extern double obsheight ;
extern float POVh;
extern float step;
extern float surScale;
extern float volScale;

//**********************************************************************************
//                   BLUR  2023
//**********************************************************************************

extern std::vector<short> pixels;
extern unsigned imgWidth,imgHeight;

//**********************************************************************************
//                   SKEWENGINE  2023  (generic)
//**********************************************************************************

extern int runMode; // 0 cpu, 1 gpu, 2 hybrid task farm en el futuro
extern color *my_palette;
extern CpuInterfaceV3 *cpu;
extern GpuInterfaceV3 *gpuV3;
extern int nthreads ;
extern int nCPUs,nGPUs;
extern int dim,dimx,dimy,N;


#endif