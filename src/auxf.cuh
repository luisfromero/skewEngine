//
// Created by Felipe on 6-12-22.
//

#ifndef TVSSDEM_AUXF_CUH
#define TVSSDEM_AUXF_CUH

#ifndef GPUSOLVER_H
#define CUDA_ERROR_CHECK
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

#ifdef _WIN32
        #include <cuda_runtime.h>
        #include <device_launch_parameters.h>
#endif


inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
#ifdef DEBUG
    #ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
            fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
            #ifdef _WIN32
            WINPAUSE;
            #endif
            exit( -1 );
    }
    #endif
#endif

    return;
}
inline void __cudaCheckError(const char *file, const int line){

#ifdef DEBUG
    #ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
            fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
            #ifdef _WIN32
            WINPAUSE;
            #endif
            exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
            fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
            #ifdef _WIN32
            WINPAUSE;
            #endif
            exit( -1 );
    }
    #endif
#endif

    return;
}
#endif

class CpuInterfaceV3
{
public:
    const int kb = 1024;
    const int mb = kb * kb;
    size_t size;
    int dimy, dimx;

    CpuInterfaceV3(int dimx, int dimy):dimx(dimx),dimy(dimy){};
    void AllocDEMHost(float *&input0,float *&input1,float *&input2,float *&input3,float *&output , int dim);
    void FreeHostMemory(float *&out, float *&input0, float *&input1, float *&input2, float *&input3);
};
class GpuInterfaceV3
{
public:

    const int kb = 1024;
    const int mb = kb * kb;
    size_t size;
    int dimy, dimx;
    /*
    float *d_DEM;
    float *d_sDEM;
    float *d_rotatedVS;
    float *d_sectorVS;
    float *d_totalVS;
    */
    GpuInterfaceV3(int dimy, int dimx);

    // Obtain current number of GPUs
    void GetNumberGPUs(int &devCount);

};


#endif //TVSSDEM_AUXF_CUH
