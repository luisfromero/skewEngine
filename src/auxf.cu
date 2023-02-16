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
void CpuInterfaceV3::AllocDEMHost(double *&heights, float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float **&h_multiSector, int devCount, float *&h_totalVS) {

    CudaSafeCall( cudaHostAlloc(&h_DEM, dimy * dimx * sizeof(*h_DEM), cudaHostAllocPortable) );
    CudaSafeCall( cudaHostAlloc(&h_sDEM, 2 * dimy * dimx * sizeof(*h_sDEM), cudaHostAllocPortable) );

    CudaSafeCall( cudaHostAlloc(&h_rotatedVS, 2 * dimy * dimx * sizeof(*h_rotatedVS), cudaHostAllocPortable) );
    CudaSafeCall( cudaHostAlloc(&h_sectorVS, dimy * dimx * sizeof(*h_sectorVS), cudaHostAllocPortable) );

    CudaSafeCall( cudaHostAlloc(&h_multiSector, devCount * sizeof(float*), cudaHostAllocPortable) );
    for (int i = 0; i < devCount; i++)
        CudaSafeCall( cudaHostAlloc(&h_multiSector[i], dimy * dimx * sizeof(float), cudaHostAllocPortable) );

    CudaSafeCall( cudaHostAlloc(&h_totalVS, dimy * dimx * sizeof(*h_totalVS), cudaHostAllocPortable) );

#ifdef DEBUG
    size = dimy * dimx * (sizeof(*h_DEM) + 2 * sizeof(*h_sDEM) + 2 * sizeof(*h_rotatedVS) + sizeof(*h_sectorVS) + sizeof(*h_totalVS));
    std::cout << "Total memory allocated in host: " << size / mb << " Mb" << std::endl;
#endif
}
void CpuInterfaceV3::FreeHostMemory(float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float **&h_multiSectorVS, int devCount, float *&h_totalVS) {

    CudaSafeCall( cudaFreeHost(h_DEM) );
    CudaSafeCall( cudaFreeHost(h_sDEM) );
    CudaSafeCall( cudaFreeHost(h_rotatedVS) );
    CudaSafeCall( cudaFreeHost(h_sectorVS) );

    for (int i = 0; i < devCount; i++)
        CudaSafeCall( cudaFreeHost(h_multiSectorVS[i]) );
    CudaSafeCall( cudaFreeHost(h_multiSectorVS) );

    CudaSafeCall( cudaFreeHost(h_totalVS) );
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

