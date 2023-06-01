//
// Created by Felipe on 11-12-22.
//

/**
 * @file main.h
 * @author Felipe Romero
 * @brief Sólo simplifica el ejecutable main
 */

// 0 Skew DEM
// 1 DEM identity (Divide by 180
// 2 Radon
// 3 Image identity (Divide by 3, and by 180
// 4 Skew blur ToDo




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
            if(filename==NULL)strcpy(fn,"4070000_0310000_010.bil");
            dimx=dimy=2000;
            N = dim = dimx * dimy;
            configureSDEM(fn);
            break;
        case 2:
            if(filename==NULL)strcpy(fn,"blurred1.png");
            configureRADN(fn);
            break;
        case 3:
            if(filename==NULL)strcpy(fn,"blurred1.png");
            configureRADN(fn);
            break;
        default:
            if(filename==NULL)strcpy(fn,"blurred1.png");
            configureRADN(fn);
            break;
    }
}


std::vector<cl::Device> OCLDevices;


void openCLcapabilities(int *nGPUs)
{
    //A platform is a specific OpenCL implementation, for instance AMD APP, NVIDIA or Intel OpenCL.
    // A context is a platform with a set of available devices for that platform.
    // And the devices are the actual processors (CPU, GPU etc.) that perform calculations.
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    auto platform = platforms[0];
    platform.getDevices(CL_DEVICE_TYPE_GPU, &OCLDevices);
    int nCLGPUS=OCLDevices.size();
    printf("Se han encontrado %d OCL-enabled GPUS\n",nCLGPUS);
    *nGPUs=nCLGPUS;

}

/**
 * Initially set n threads to ncores and n gpus to available gpus
 * @param dimx
 * @param dimy
 * @param runMode
 */
void setResources(int dimx,int dimy, int runMode=CPU_MODE, int gpuMode=CUDA_MODE)
{
    nGPUs = 0;
    if(gpuMode==CUDA_MODE)
        cudaGetDeviceCount(&nGPUs);
    if(gpuMode==OPENCL_MODE)
        openCLcapabilities(&nGPUs);

    // OpenCL
    nCPUs=omp_get_num_procs();
    if(runMode==CPU_MODE)nGPUs=0;
    if(nthreads==-1)
    {
        nthreads=nCPUs;
        while((180%nthreads))nthreads--;
        printf("Now, nthreads set to %d\n",nthreads);
    }
    if(runMode==GPU_MODE)nthreads=nGPUs;
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



/**
 * Switch que selecciona la forma de mostrar los resultados, dependiendo del kernel
 * @param skewAlgorithm
 */
void showResults(int skewAlgorithm) {
    pair_t mm = helper::getMinMax(outD);
    float escala = 1.0 / 180;
    float autoScale = 255/(mm.max-mm.min);
    if (skewAlgorithm == 0)escala = M_PI/(360*10*10); //scales to hectometers
    if (skewAlgorithm == 1)escala = 1/180.0; //scales to max
    if (skewAlgorithm == 2)escala = autoScale;//1.0/180;  //radon
    if (skewAlgorithm == 3)escala = 1.0 / 540; //identity blur

    printf("Extreme values (unscaled) for output: %6.2f | %e  \n ", mm.min , mm.max );
    printf("Extreme values for output: %6.2f | %e  (scale = %f )\n ", (mm.min * escala), mm.max * escala, escala);
    fflush(stdout);

    if(skewAlgorithm==0)showResultsSDEM();
    if(skewAlgorithm==1)showResultsSDEM();
    if(skewAlgorithm==2)showResultsRADN(escala,mm.min);
    if(skewAlgorithm==3)showResultsRADN(escala);

}
