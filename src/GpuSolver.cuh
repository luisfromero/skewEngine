#ifndef GPUSOLVER_H
#define GPUSOLVER_H


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

#ifdef DEBUG
//#include "nvToolsExt.h"
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
//#include "Global.h"
#include "omp.h"


class CpuInterface
{
public:

    const int kb = 1024;
    const int mb = kb * kb;
    size_t size;

    int dimy, dimx;

    // Constructors
    CpuInterface();
    CpuInterface(int dimy, int dimx);

    // Allocate host data   
    void AllocDEMHost(double *&heights, float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float *&h_totalVS);
    void AllocDEMHost(double *&heights, float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float **&h_multiSectorVS, int devCount, float *&h_totalVS);

    // Free host memory
    void FreeHostMemory(float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float *&h_totalVS);
    void FreeHostMemory(float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float **&h_multiSectorVS, int devCount, float *&h_totalVS);
};


// Class CUDA file
class GpuInterface
{
public:
    
    const int kb = 1024;
    const int mb = kb * kb;
    size_t size;

    int dimy, dimx;
    int devIndex;

    float *d_DEM;
    float *d_sDEM;
    float *d_rotatedVS;
    float *d_sectorVS;
    float *d_totalVS;
    
    // Constructors
    GpuInterface();
    GpuInterface(int dimy, int dimx);
    GpuInterface(int dimy, int dimx, int deviceIndex);

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


#endif