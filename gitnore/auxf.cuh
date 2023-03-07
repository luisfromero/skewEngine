/**
 * @file
 * @author Felipe Romero
 * @brief Aparentemente está pensado para eliminar Gpusolver
 */

#ifndef AUXF_CUH
#define AUXF_CUH

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


#if defined(WIN32)
#define WINPAUSE system("pause")
#else
#define WINPAUSE
#endif





template <typename T>
class CpuInterface
{
public:
    int dimy, dimx;
    CpuInterface<T>(int dimx, int dimy):dimx(dimx),dimy(dimy){};
    void AllocDEMHost(T *&input0,T *&input1,T *&input2,T *&input3, int dim);
    void FreeHostMemory(T *&input0, T *&input1, T *&input2, T *&input3);
};


#endif //TVSSDEM_AUXF_CUH
