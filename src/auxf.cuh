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
    void AllocDEMHost(double *&heights, float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float **&h_multiSectorVS, int devCount, float *&h_totalVS);
    void FreeHostMemory(float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float **&h_multiSectorVS, int devCount, float *&h_totalVS);
    void FreeHostMemory(float *&out, float *&input0, float *&input1, float *&input2, float *&input3);
};
class GpuInterfaceV3
{
public:

    const int kb = 1024;
    const int mb = kb * kb;
    size_t size;
    int dimy, dimx;

    float *d_DEM;
    float *d_sDEM;
    float *d_rotatedVS;
    float *d_sectorVS;
    float *d_totalVS;

    GpuInterfaceV3(int dimy, int dimx);

    // Show device properties
    void DeviceProperties();

    // Allocate device data
    void AllocDEMDevice(int deviceIndex);

    // Copy data from host to device
    void MemcpyDEM_H2D(float *&h_DEM, int deviceIndex);

    // Main operations
    void Execute(float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float *&h_totalVS, int angle, int startGPUbatch, int endGPUbatch, int deviceIndex, float POVh);

    // Calculate non-rotated viewshed (only for single GPU)
    void CalculateVS(int angle);

    // Multi GPU execution
    // void ExecuteMultiGPUopenmp(float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float *&h_totalVS, int angle);
    void ExecuteSingleSectorMultiGPUmaster(float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float *&h_totalVS, int angle, int devCount, float POVh);
    void ExecuteAccMultiGPUmaster(float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float **&h_multiSectorVS, float *&h_totalVS, int maxAngle, int devCount, float POVh);

    // Copy data from device to host
    void MemcpyDEM_D2H(float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float *&h_totalVS);
    void MemcpyDEM_D2Hheterogeneous(float *&h_rotatedVS, int startGPUbatch, int endGPUbatch, int deviceIndex);

    // Wait for device to end
    void Syncronize(int deviceIndex);

    // Obtain current number of GPUs
    void GetNumberGPUs(int &devCount);

    // Free device memory
    void FreeDeviceMemory();
};


#endif //TVSSDEM_AUXF_CUH
