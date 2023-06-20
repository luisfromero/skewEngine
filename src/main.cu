#include <cstdio>
#include <fstream>
#include <omp.h>

#define WIN32_LEAN_AND_MEAN

#define USE_OPENCL
#ifdef USE_OPENCL
#include <CL/cl.hpp>
#endif

#include <algorithm>
#ifdef _WIN32
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
// 1 DEM identity (Divide by 180
// 2 Radon?
// 3 Image identity (Divide by 3, and by 180
// 4 Skew blur

/**
 * Algoritmo que se implementa (totalviewshed, transformada radon, identidades, ...)
 */
int skewAlgorithm;
bool timing=true;


#include "kernelSDEM.h"
#include "kernelRADN.h"
#include "kernelBLUR.h"
#include "kernelUNIT.h"
#include "main.h"



void execute(int skewAlgorithm)
{
//inData is the input data
//inputD is a struct with 4 pointers (4 arrays of input data: NN, TM, TN, TM)

//    inputData<float> inData=  skewEngine<float>::prepare(inputD,dimx,dimy);//Create rotated and mirror versions of input
    //bool ident=skewAlgorithm==2||skewAlgorithm==3;
    //omp_set_num_threads(1);
    samplePoints.push_back(punto1);
    samplePoints.push_back(punto2);
    samplePoints.push_back(punto3);

    for (int i = 0; i < maxSector ; i++)allocSampleData(i);


#pragma omp parallel default(none) shared(inData,dimx,dimy,runMode,gpuMode,OCLDevices,outD,maxSector,skewAlgorithm,timing)
    {
        int id = omp_get_thread_num();

// Each thread (in CPU mode -> arbitrary)  (in GPU mode -> nthreads = num of GPUs) has its own engine:
        skewEngine<float> *skewer=new skewEngine<float>(dimx, dimy, static_cast<inputData<float>>(inData), runMode == GPU_MODE,id);
        //Dentro del constructor? ->
        cl::Context OCLContext;
        cl::Device OCLDevice;
        if(runMode==GPU_MODE && gpuMode == OPENCL_MODE) {
            skewer->isCUDA=false;
            OCLDevice=OCLDevices[id];
            skewer->OCLDevice =OCLDevice;
            OCLContext=cl::Context(OCLDevice);
            skewer->OCLContext = OCLContext;
            cl::CommandQueue OCLQueue;
            OCLQueue= cl::CommandQueue(OCLContext,OCLDevice);
            skewer->OCLQueue = OCLQueue;
        }


        switch(skewAlgorithm) {
            case KERNEL_SDEM:
                skewer->kernelcpu = kernelV3;
                skewer->kernelcuda =kernelV3cuda;
                skewer->kernelOCL = kernelV3OCL;
                break;
            case KERNEL_RADON:
                cufftHandle plan, cb_plan;
                size_t work_size;
                cufftCreate(&plan);
                cufftCreate(&cb_plan);
                cufftComplex *data;
                cufftMakePlan1d(reinterpret_cast<cufftHandle>(&plan), sizeof(cufftComplex) * skewer->skewHeight, CUFFT_R2C, 1, &work_size);
                skewer->kernelcpu = radon;
                skewer->kernelcuda =kernelRadonCuda;
                skewer->kernelOCL = kernelRadonOCL;
                skewer->lineCUDA=true;
                break;

            case KERNEL_CEPSTRUM:
                // Acumula
                skewer->kernelcpu = cepstrum;
                //ToDo Versión cepstrum  para CUDA
                break;
            default: // Casos 1 y 3
                skewer->kernelcpu = identity;
                skewer->kernelcuda = identityCuda;
                skewer->kernelOCL = static_cast<const std::basic_string<char>>(identityOCL);
                break;
        }



        skewer->skewAlloc();




#pragma omp barrier
#pragma omp for schedule(dynamic) nowait
        for (int i = 0; i < maxSector; i++) {


            skewer->skew(i);


            skewer->kernel();


            skewer->deskew(skewAlgorithm==4?   1:0);
            if(!timing)printf("id= %03d se=%03d\n",id,i);//fflush(stdout);
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
    skewAlgorithm=KERNEL_SDEM;
    //skewAlgorithm=KERNEL_IDENTITYDEM;
    runMode=CPU_MODE;
    runMode=GPU_MODE;
    gpuMode=CUDA_MODE;
    gpuMode=OPENCL_MODE;
    configure(argc, argv); // Read input data (model, image...) and parameters
    setResources(dimx,dimy,runMode,gpuMode); // Create cpu and gpu interfaces, set nCPUs, nGPUs
    //Allocate es static. El objeto aún no existe
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
