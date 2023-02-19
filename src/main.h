//
// Created by Felipe on 11-12-22.
//

/**
 * @file main.h
 * @author Felipe Romero
 * @brief Sólo simplifica el ejecutable main
 */

// 0 Skew DEM
// 1 Skew blur (max)
// 2 DEM identity (Divide by 180
// 3 Image identity (Divide by 3, and by 180
// 4 Skew blur



/**
 * Switch para configurar la ejecución de cada uno de los kernels
 * @param argc
 * @param argv
 * @param filename
 */
void configure(int argc, char *argv[], char * filename=NULL) {
    char fn[100];
    if(filename!=NULL)strcpy(fn,filename);
    switch(skewAlgorithm)
    {
        case 0:
            if(filename==NULL)strcpy(fn,"4070000_0310000_010.bil");
            dimx=dimy=2000;
            N = dim = dimx * dimy;
            configureSDEM(fn);
            break;
        case 1:
            if(filename==NULL)strcpy(fn,"blurred1.png");
            configureRADN(fn);
            break;
        case 2:
            if(filename==NULL)strcpy(fn,"4070000_0310000_010.bil");
            dimx=dimy=2000;
            N = dim = dimx * dimy;            configureSDEM(filename);
            configureSDEM(fn);
            break;
        case 3:
            if(filename==NULL)strcpy(fn,"blurred1.png");
            configureRADN(fn);
            break;
        case 4:
            if(filename==NULL)strcpy(fn,"blurred1.png");
            configureRADN(fn);
            break;
        default:
            if(filename==NULL)strcpy(fn,"blurred1.png");
            configureRADN(fn);
            break;
    }
}

/**
 * Initially set n threads to ncores and n gpus to available gpus
 * @param dimx
 * @param dimy
 * @param runMode
 */
void setResources(int dimx,int dimy, int runMode=0)
{
    cpu = new CpuInterfaceV3(dimy, dimx);
    gpuV3 = new GpuInterfaceV3(dimy, dimx);
    nGPUs = 0;
    gpuV3->GetNumberGPUs(nGPUs);
    nCPUs=omp_get_num_procs();
    if(runMode==0)nGPUs=0;
    if(nthreads==-1)
    {
        nthreads=nCPUs;
        while((180%nthreads))nthreads--;
        printf("Now, nthreads set to %d\n",nthreads);
    }
    if(runMode==1)nthreads=nGPUs;
    //What happens in runmode 2? Not set
    omp_set_num_threads(nthreads);
    if(verbose) {
        printf("%d CPUs and %d GPUs found. nthreads set to %d\n", nCPUs, nGPUs, nthreads);
        printf("Allocating DEM (size: %dx%d)\n", dimx, dimy);
    }

}

/**
 * Ejecución del skewEngine
 * @param skewAlgorithm Algoritmo que se implementa (totalviewshed, transformada radon, identidades, ...)
 */
void execute(int skewAlgorithm=0);



void showResults(int skewAlgorithm) {
    pair_t mm = getMinMax(outD);
    float escala = 1.0 / 180;
    if (skewAlgorithm == 0)escala = surScale; //scales to hectometers
    if (skewAlgorithm == 1)escala = 1;//1.0/180;  //cepstrum
    if (skewAlgorithm == 3)escala = 1.0 / 540; //identity blur

    printf("Extreme values for output: %6.2f - %e  (scale = %f )\n ", (mm.min * escala), mm.max * escala, escala);
    fflush(stdout);

    if(skewAlgorithm==0)showResultsSDEM();
    if(skewAlgorithm==1)showResultsRADN();
    if(skewAlgorithm==3)showResultsRADN();



}
