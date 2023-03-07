#include <cstdio>
#include <fstream>
#include <omp.h>
#include <algorithm>
#ifdef WIN32
#include <filesystem>
namespace fs = std::filesystem;
#define I_DIR "d:/input/"
#define O_DIR "d:/output/"
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#define I_DIR "/users/felipe/input/"
#define O_DIR "/users/felipe/output/"
#endif


#include "helper.h"
//#include "../gitnore/auxf.h"  //<< global.f << defaults.h
//#include "auxf.cuh"


// 0 Skew DEM
// 1 Skew blur (max)
// 2 DEM identity (Divide by 180
// 3 Image identity (Divide by 3, and by 180
// 4 Skew blur

/**
 * Algoritmo que se implementa (totalviewshed, transformada radon, identidades, ...)
 */
int skewAlgorithm;


#include "kernelSDEM/kernelSDEM.h"
#include "kernelRADN/kernelRADN.h"
#include "kernelBLUR/kernelBLUR.h"
#include "kernelUNIT/kernelUNIT.h"
#include "main.h"



void execute(int skewAlgorithm)
{
//inData is the input data
//inputD is a struct with 4 pointers (4 arrays of input data: NN, TM, TN, TM)

//    inputData<float> inData=  skewEngine<float>::prepare(inputD,dimx,dimy);//Create rotated and mirror versions of input
    bool ident=skewAlgorithm==2||skewAlgorithm==3;
    //omp_set_num_threads(1);
    samplePoints.push_back(punto1);
    samplePoints.push_back(punto2);
    samplePoints.push_back(punto3);

    for (int i = 0; i < 180; i++)allocSampleData(i);


#pragma omp parallel default(none) shared(inData,dimx,dimy,runMode,outD,maxSector,ident,skewAlgorithm)
    {
        int id = omp_get_thread_num();

// Each thread (in CPU mode -> arbitrary)  (in GPU mode -> nthreads = num of GPUs) has its own engine:
        skewEngine<float> *skewer=new skewEngine<float>(dimx, dimy, static_cast<inputData<float>>(inData), runMode == 1,id);
#pragma omp barrier
#pragma omp for schedule(dynamic) nowait
        for (int i = 0; i < maxSector; i++) {


            skewer->skew(i);
            switch(skewAlgorithm) {
                case 0:
                    skewer->kernelcpu = kernelV3;
                    skewer->kernelgpu =kernelV3cuda;
                    break;
                case 2:
                    // Selecciono el máximo
                    skewer->kernelcpu = radon;
                    //ToDo Versión blur para CUDA
                    break;
                case 4:
                    // Acumula
                    skewer->kernelcpu = cepstrum;
                    //ToDo Versión blur para CUDA
                    break;
                default:
                    skewer->kernelcpu = identity;
                    skewer->kernelgpu =identityCuda;
                    break;
            }


            skewer->kernel();

            skewer->deskew(skewAlgorithm==4?   1:0);
            printf("id= %03d se=%03d\n",id,i);//fflush(stdout);
        }

#pragma omp critical
        {
            // When finishing, thread data are added to outD
            skewer->reduce(outD,skewAlgorithm==4?1:0);
        }
        delete skewer;



    } //end parallel region



}

int main(int argc, char *argv[]) {
    fs::path p=fs::current_path();
    skewAlgorithm=2;
    runMode=0;
    configure(argc, argv); // Read input data (model, image...) and parameters
    setResources(dimx,dimy,runMode); // Create cpu and gpu interfaces, set nCPUs, nGPUs
    skewEngine<float>::allocate((inputData<float> &) inData, inputD,  (float **) & outD, dimx, dimy);
    std::chrono::time_point<std::chrono::high_resolution_clock> t1 = std::chrono::high_resolution_clock::now();
    //omp_set_num_threads(1);
    execute(skewAlgorithm);
    std::chrono::time_point<std::chrono::high_resolution_clock> t2 = std::chrono::high_resolution_clock::now();
    showResults(skewAlgorithm);
    double t = (double)(t2 - t1).count() / 1000000000;printf("Tiempo: %f\n",t);
    skewEngine<float>::deallocate(inData,inputD,outD);
    return 0;
}

/**
 * Interfaz dll
 */
extern "C" int run(int argc, char *argv[]){
    return main(argc,argv);
}
