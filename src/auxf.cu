//
// Created by Felipe on 6-12-22.
//
#include "auxf.cuh"

void CpuInterfaceV3::AllocDEMHost(float *&outputD, float *&input0, float *&input1, float *&input2, float *&input3, int dim) {
    CudaSafeCall( cudaHostAlloc(&input0, dim* sizeof(*input0), cudaHostAllocPortable) );
    CudaSafeCall( cudaHostAlloc(&input1, dim* sizeof(*input1), cudaHostAllocPortable) );
    CudaSafeCall( cudaHostAlloc(&input2, dim* sizeof(*input2), cudaHostAllocPortable) );
    CudaSafeCall( cudaHostAlloc(&input3, dim* sizeof(*input3), cudaHostAllocPortable) );
    CudaSafeCall( cudaHostAlloc(&outputD, dim * sizeof(*outputD), cudaHostAllocPortable) );
}

void CpuInterfaceV3::FreeHostMemory(float *&out, float *&input0, float *&input1, float *&input2, float *&input3 ) {

    CudaSafeCall( cudaFreeHost(out) );
    CudaSafeCall( cudaFreeHost(input0) );
    CudaSafeCall( cudaFreeHost(input1) );
    CudaSafeCall( cudaFreeHost(input2) );
    CudaSafeCall( cudaFreeHost(input3) );
}

GpuInterfaceV3::GpuInterfaceV3(int dimy, int dimx) {

    this->dimy = dimy;
    this->dimx = dimx;
}

void GpuInterfaceV3::GetNumberGPUs(int &devCount) {

    cudaGetDeviceCount(&devCount);
}

