//
// Created by Felipe on 11-12-22.
//
#include <cstdio>
#include <iostream>
#include <string>
#include <complex>
#include <valarray>

#include "auxf.h"  //<< global.f << defaults.h
#include "auxf.cuh"
#include "Lodepng.h"
#include "drawing.h"




// Set n threads, n gpus
void setResources(int dimx,int dimy, int runMode=0)
{
    cpu = new CpuInterfaceV3(dimy, dimx);
    gpuV3 = new GpuInterfaceV3(dimy, dimx);
    nGPUs = 0;
    gpuV3->GetNumberGPUs(nGPUs);
    nCPUs=omp_get_num_procs();
    if(nthreads==-1)
    {
        nthreads=nCPUs;
        while((180%nthreads))nthreads--;
        printf("Now, nthreads set to %d\n",nthreads);
    }
    if(runMode==1)nthreads=nGPUs;
    //What happens in runmode 2?
    omp_set_num_threads(nthreads);
    if(verbose) {
        printf("%d CPUs and %d GPUs found. nthreads set to %d\n", nCPUs, nGPUs, nthreads);
        printf("Allocating DEM (size: %dx%d)\n", dimx, dimy);
    }

}

template <typename T>
void readData(char *inputfilename, T *&inputData)
{
    //Uncomment for a simple,non-square input test:
    dimy=1980;dimx=1920;dim=dimx*dimy;

    FILE *f;
    f = fopen(inputfilename, "rb");
    if (f == NULL) {
        printf("Error opening %s\n", inputfilename);
    }
    else {
        for (int i = 0; i < dimy; i++) {
            for (int j = 0; j < dimx; j++) {
                short num;
                fread(&num, 2, 1, f);
                inputData[dimx * i + j] = ((T) num) / 10.0; //internal representation from top to bottom (inner loop)
            }
            //simple non-square input test:
            fseek(f, 2*(2000-dimx), SEEK_CUR);
        }
        fclose(f);
        pair_t mm= getMinMax(inputData);
        printf("Input model readed, with extreme values (/step): %5.1f - %6.1f\n",mm.min*step,mm.max*step);
    }
}

void configureSDEM(char *filename)
{
    inputD=h_DEM = new float[dim];
    readData(filename, inputD);
    surScale=M_PI/(360*step*step);
    POVh=obsheight/step;
    if(verbose) {
        printf("Allocating DEM (from %s, with filesize: %dx%d and step %f) and setting observer's height to %f\n", filename, dimx, dimy,step,obsheight);
    }
}

void configureBLUR(char *filename) {
    read_png(filename,pixels,imgWidth,imgHeight);
    dimx=imgWidth;
    dimy=imgHeight;
    dim=N=dimx*dimy;
    inputD=new float[dim];
    for(int i=0;i<dim;i++)inputD[i]=pixels[i];
}


int skewAlgorithm=0;
void configureV3(int argc, char *argv[], char * filename) {
    switch(skewAlgorithm)
    {
        case 0:
            check_args_v2(argc,argv);
            configureSDEM(filename);
            break;
        case 1:
            configureBLUR(filename);
            break;
        case 2:
            check_args_v2(argc,argv);
            configureSDEM(filename);
            break;
        case 3:
            configureBLUR(filename);
            break;
        case 4:
            configureBLUR(filename);
            break;
        default:
            check_args_v2(argc,argv);
            configureSDEM(filename);
            break;
    }

}
