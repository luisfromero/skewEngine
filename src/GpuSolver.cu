#include "GpuSolver.cuh"

#if defined(WIN32)
#define WINPAUSE system("pause")
#else
#include <unistd.h>
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


// KERNEL FOR DEBUGGING (NOT USED ANYMORE)
__global__
void sequentialRotation(float *sDEM, float *heights, int *desti, float *rat, int dimy, int dimx) {

    for (int i = 0; i < dimy; i++)
        for (int j = 0; j < dimx; j++) {
            sDEM[ (dimy + i - desti[j]) * dimy + j ] += (1.0 - rat[j]) * heights[dimx * i + j];
            sDEM[ (dimy + i - desti[j] - 1) * dimy + j] += rat[j] * heights[dimx * i + j];
        }
}


// KERNEL FOR DEBUGGING (NOT USED ANYMORE)
__global__
void rotatedViewshedCalculation(float *sDEM, float *rotatedVS, int angle, int dimy, int dimx) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int strideRow = gridDim.x * blockDim.x;

    int ky = tanf(angle * M_PI / 180) * (dimx - 1);
    row += dimy - ky - 1;

    int filafinal = 2 * dimy;

    float r = (1.0 / cos(angle * M_PI / 180.0)) * (1.0 / cos(angle * M_PI / 180.0));

    for (int i = row; i < filafinal; i += strideRow) {                                 // Calcular for la cuenca visual de i (a izqda y a dcha)

        int cnt = 0;                                                                    
        bool found = false, lock = false;
        int start = 0, end = 0;   
        for (int j = 0; j < dimx; j++) {
                
            found = (sDEM[i * dimy + j] > 0) ? 1 : 0;
            if (found && !lock) {
                lock = true;
                start = j;
            }
            if (found) end = j + 1;
            cnt += (sDEM[i * dimy + j] > 0) ? 1 : 0;
        }

        for (int j = start; j < end; j++) {

            float cv = 0;                                                               // Ya tengo observador en j
            double h = sDEM[i * dimy + j];
            float max_angle = -2000;
            bool visible = true;
            float open_delta_d = 0;
            for (int k = j - 1; k >= start; k--) {                                      // Calculo cv a izqda

                float delta_d = j - k;
                float angle = (sDEM[i * dimy + k] - h) / delta_d;
                bool above = (angle > max_angle);
                bool opening = above && (!visible);
                bool closing = (!above) && (visible);

                visible = above;
                max_angle = max(angle, max_angle);
                if (opening) open_delta_d = delta_d;
                if (closing) cv += (delta_d * delta_d - open_delta_d * open_delta_d);
            }

            max_angle = -2000;
            visible = true;
            open_delta_d = 0;
            for (int k = j + 1; k < end; k++) {                                         // Calculo cv a dcha

                float delta_d = k - j;
                float angle = (sDEM[i * dimy + k] - h) / delta_d;
                bool above = (angle > max_angle);
                bool opening = above && (!visible);
                bool closing = (!above) && (visible);

                visible = above;
                max_angle = max(angle, max_angle);
                if (opening) open_delta_d = delta_d;
                if (closing) cv += (delta_d * delta_d - open_delta_d * open_delta_d);
            }

            rotatedVS[i * dimy + j] = cv * r;
            // if (i >= 1000 && i < 1001 && j == start) printf("[%d][%d][start:%d,end:%d\n", i, j, start, end);
        }

    }

}


// KERNEL-1.1 FROM 0 TO 45 DEGREES
__global__
void optimizedRotation0to45(float *sDEM, float *heights, int angle, int dimy, int dimx) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float y = tanf(angle * M_PI / 180.0) * j;
    int dest = y;
    float r = y - dest;

    if (i < dimy && j < dimx) {
        atomicAdd(&sDEM[ (dimy + i - dest) * dimy + j ], (1.0 - r) * heights[dimx * i + j]);
        atomicAdd(&sDEM[ (dimy + i - dest - 1) * dimy + j ], r * heights[dimx * i + j]);
    }
}   


// KERNEL-1.2 FROM 46 TO 90 DEGREES
__global__
void optimizedRotation45to90(float *sDEM, float *heights, int angle, int dimy, int dimx) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float y = tanf((90 - angle) * M_PI / 180.0) * i;
    int dest = y;
    float r = y - dest;

    if (i < dimy && j < dimx) {
        atomicAdd(&sDEM[ (dimy + j - dest) * dimy + i ], (1.0 - r) * heights[dimx * i + j]);
        atomicAdd(&sDEM[ (dimy + j - dest - 1) * dimy + i ], r * heights[dimx * i + j]);
    }   
}   


// KERNEL-1.3 FROM 91 TO 135 DEGREES
__global__
void optimizedRotation90to135(float *sDEM, float *heights, int angle, int dimy, int dimx) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float y = tanf((angle - 90) * M_PI / 180.0) * i;
    int dest = y;
    float r = y - dest;

    if (i < dimy && j < dimx) {
        atomicAdd(&sDEM[ (j + dest) * dimy + i ], (1.0 - r) * heights[dimx * i + j]);
        atomicAdd(&sDEM[ (j + dest + 1) * dimy + i ], r * heights[dimx * i + j]);
    }   
}   


// KERNEL-1.4 FROM 136 TO 180 DEGREES
__global__
void optimizedRotation135to180(float *sDEM, float *heights, int angle, int dimy, int dimx) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float y = tanf((180 - angle) * M_PI / 180.0) * j;
    int dest = y;
    float r = y - dest;

    if (i < dimy && j < dimx) {
        atomicAdd(&sDEM[ (i + dest) * dimy + j ], (1.0 - r) * heights[dimx * i + j]);
        atomicAdd(&sDEM[ (i + dest + 1) * dimy + j ], r * heights[dimx * i + j]);
    }   
}   


// KERNEL-2.1 FROM 0 TO 45 DEGREES
__global__
void rotatedVSComputation0to45(float *sDEM, float *rotatedVS, int angle, int dimy, int dimx, int startGPUbatch, int endGPUbatch, float POVh) {

    int i = blockIdx.x * blockDim.x + threadIdx.x + startGPUbatch;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    float r = (1.0 / cos(angle * M_PI / 180.0)) * (1.0 / cos(angle * M_PI / 180.0));

    float tanAngComp = tanf((90 - angle) * M_PI / 180);
    float tanAng = tanf(angle * M_PI / 180);
    int offset = dimy - (dimx - 1) * tanAng;
    int threshold = offset + dimy;
    int filafinal = 2 * dimy;

    // if (i == 0 && j == 0) printf("off:%d - th:%d\n", offset, threshold);

    int start, end;
    float cv, h, max_angle, open_delta_d, delta_d, cAngle;
    bool visible, above, opening, closing;

    if (i < endGPUbatch && j < dimx) {

        start = 0;
        end = 0;

        if (i >= offset && i < dimy) {
            start = (dimx - 1) - (i - offset) * tanAngComp; 
            end = dimx;
        }
        else if (i >= dimy && i < threshold) {
            start = 0;
            end = dimx;
        }
        else if (i >= threshold && i < filafinal) {
            start = 0;
            end = dimx - (i - dimy - offset) * tanAngComp; 
        }

        cv = 0;                                                                                 // Ya tengo observador en j
        h = sDEM[i * dimy + j] + POVh;
        max_angle = -2000;
        visible = true;
        open_delta_d = 0;

        for (int k = j - 1; k >= start && k < end; k--) {                                      // Calculo cv a izqda

            delta_d = j - k;
            cAngle = (sDEM[i * dimy + k] - h) / delta_d;
            above = (cAngle > max_angle);
            opening = above && (!visible);
            closing = (!above) && (visible);

            visible = above;
            max_angle = max(cAngle, max_angle);
            if (opening) open_delta_d = delta_d;
            if (closing) cv += (delta_d * delta_d - open_delta_d * open_delta_d);
        }

        max_angle = -2000;
        visible = true;
        open_delta_d = 0;

        for (int k = j + 1; k >= start && k < end; k++) {                                       // Calculo cv a dcha

            delta_d = k - j;
            cAngle = (sDEM[i * dimy + k] - h) / delta_d;
            above = (cAngle > max_angle);
            opening = above && (!visible);
            closing = (!above) && (visible);

            visible = above;
            max_angle = max(cAngle, max_angle);
            if (opening) open_delta_d = delta_d;
            if (closing) cv += (delta_d * delta_d - open_delta_d * open_delta_d);
        }

        rotatedVS[i * dimy + j] = cv * r;
    }

}


// KERNEL-2.3 FROM 91 TO 135 DEGREES
__global__
void rotatedVSComputation90to135(float *sDEM, float *rotatedVS, int angle, int dimy, int dimx, int startGPUbatch, int endGPUbatch, float POVh) {

    int i = blockIdx.x * blockDim.x + threadIdx.x + startGPUbatch;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    float r = (1.0 / cos((angle - 90) * M_PI / 180.0)) * (1.0 / cos((angle - 90) * M_PI / 180.0));

    float tanAngComp = tanf((angle - 90) * M_PI / 180);
    float tanAng = tanf((180 - angle) * M_PI / 180);
    int offset = (dimx - 1) * tanAngComp;
    int threshold = offset + dimy;
    int filafinal = threshold;

    // if (i == 0 && j == 0) printf("off:%d - th:%d\n", offset, threshold);

    int start, end;
    float cv, h, max_angle, open_delta_d, delta_d, cAngle;
    bool visible, above, opening, closing;

    if (i < filafinal && j < dimx) {

        start = 0;
        end = 0;

        if (i >= 0 && i < offset) {
            start = 0;
            end = i * tanAng;
        }
        else if (i >= offset && i < dimy) {
            start = 0;
            end = dimx;
        }
        else if (i >= dimy && i < threshold) {
            start = (i - (dimy - 1)) * tanAng;
            end = dimx;
        }

        cv = 0;                                                                                 // Ya tengo observador en j
        h = sDEM[i * dimy + j] + POVh;
        max_angle = -2000;
        visible = true;
        open_delta_d = 0;

        for (int k = j - 1; k >= start && k < end; k--) {                                      // Calculo cv a izqda

            delta_d = j - k;
            cAngle = (sDEM[i * dimy + k] - h) / delta_d;
            above = (cAngle > max_angle);
            opening = above && (!visible);
            closing = (!above) && (visible);

            visible = above;
            max_angle = max(cAngle, max_angle);
            if (opening) open_delta_d = delta_d;
            if (closing) cv += (delta_d * delta_d - open_delta_d * open_delta_d);
        }

        max_angle = -2000;
        visible = true;
        open_delta_d = 0;

        for (int k = j + 1; k >= start && k < end; k++) {                                       // Calculo cv a dcha

            delta_d = k - j;
            cAngle = (sDEM[i * dimy + k] - h) / delta_d;
            above = (cAngle > max_angle);
            opening = above && (!visible);
            closing = (!above) && (visible);

            visible = above;
            max_angle = max(cAngle, max_angle);
            if (opening) open_delta_d = delta_d;
            if (closing) cv += (delta_d * delta_d - open_delta_d * open_delta_d);
        }

        rotatedVS[i * dimy + j] = cv * r;
    }

}


// KERNEL-2.4 FROM 136 TO 180 DEGREES
__global__
void rotatedVSComputation135to180(float *sDEM, float *rotatedVS, int angle, int dimy, int dimx, int startGPUbatch, int endGPUbatch, float POVh) {

    int i = blockIdx.x * blockDim.x + threadIdx.x + startGPUbatch;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    float r = (1.0 / cos((180 - angle) * M_PI / 180.0)) * (1.0 / cos((180 - angle) * M_PI / 180.0));

    float tanAngComp = tanf((90 - (180 - angle)) * M_PI / 180);
    float tanAng = tanf((180 - angle) * M_PI / 180);
    int offset = (dimx - 1) * tanAng;
    int threshold = offset + dimy;
    int filafinal = threshold;

    // if (i == 0 && j == 0) printf("off:%d - th:%d\n", offset, threshold);

    int start, end;
    float cv, h, max_angle, open_delta_d, delta_d, cAngle;
    bool visible, above, opening, closing;

    if (i < filafinal && j < dimx) {

        start = 0;
        end = 0;

        if (i >= 0 && i < offset) {
            start = 0;
            end = i * tanAngComp;
        }
        else if (i >= offset && i < dimy) {
            start = 0;
            end = dimx;
        }
        else if (i >= dimy && i <= threshold) {
            start = (i - (dimy - 1)) * tanAngComp;
            end = dimx;
        }

        cv = 0;                                                                                 // Ya tengo observador en j
        h = sDEM[i * dimy + j] + POVh;
        max_angle = -2000;
        visible = true;
        open_delta_d = 0;

        for (int k = j - 1; k >= start && k < end; k--) {                                      // Calculo cv a izqda

            delta_d = j - k;
            cAngle = (sDEM[i * dimy + k] - h) / delta_d;
            above = (cAngle > max_angle);
            opening = above && (!visible);
            closing = (!above) && (visible);

            visible = above;
            max_angle = max(cAngle, max_angle);
            if (opening) open_delta_d = delta_d;
            if (closing) cv += (delta_d * delta_d - open_delta_d * open_delta_d);
        }

        max_angle = -2000;
        visible = true;
        open_delta_d = 0;

        for (int k = j + 1; k >= start && k < end; k++) {                                       // Calculo cv a dcha

            delta_d = k - j;
            cAngle = (sDEM[i * dimy + k] - h) / delta_d;
            above = (cAngle > max_angle);
            opening = above && (!visible);
            closing = (!above) && (visible);

            visible = above;
            max_angle = max(cAngle, max_angle);
            if (opening) open_delta_d = delta_d;
            if (closing) cv += (delta_d * delta_d - open_delta_d * open_delta_d);
        }

        rotatedVS[i * dimy + j] = cv * r;
    }

}


// KERNEL-3.1 FROM 0 TO 45 DEGREES
__global__
void viewshedCalculation0to45(float *sectorVS, float *rotatedVS, int angle, int dimy, int dimx) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float y = tanf(angle * M_PI / 180.0) * j;
    int dest = y;
    float r = y - dest;

    if (i < dimy && j < dimx)
        sectorVS[i * dimy + j] += (1.0 - r) * rotatedVS[(dimy + i - dest) * dimy + j] + r * rotatedVS[(dimy + i - dest - 1) * dimy + j];		// WARNING: ACCUMULATED
}


// KERNEL-3.2 FROM 46 TO 90 DEGREES
__global__
void viewshedCalculation45to90(float *sectorVS, float *rotatedVS, int angle, int dimy, int dimx) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float y = tanf((90 - angle) * M_PI / 180.0) * i;
    int dest = y;
    float r = y - dest;

    if (i < dimy && j < dimx)
        sectorVS[i * dimy + j] += (1.0 - r) * rotatedVS[(dimy + j - dest) * dimy + i] + r * rotatedVS[(dimy + j - dest - 1) * dimy + i];		// WARNING: ACCUMULATED
}


// KERNEL-3.3 FROM 91 TO 135 DEGREES
__global__
void viewshedCalculation90to135(float *sectorVS, float *rotatedVS, int angle, int dimy, int dimx) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float y = tanf((angle - 90) * M_PI / 180.0) * i;
    int dest = y;
    float r = y - dest;

    if (i < dimy && j < dimx)
        sectorVS[i * dimy + j] += (1.0 - r) * rotatedVS[(j + dest) * dimy + i] + r * rotatedVS[(j + dest + 1) * dimy + i];		// WARNING: ACCUMULATED
}


// KERNEL-3.4 FROM 136 TO 180 DEGREES
__global__
void viewshedCalculation135to180(float *sectorVS, float *rotatedVS, int angle, int dimy, int dimx) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float y = tanf((180 - angle) * M_PI / 180.0) * j;
    int dest = y;
    float r = y - dest;

    if (i < dimy && j < dimx)
        sectorVS[i * dimy + j] += (1.0 - r) * rotatedVS[(i + dest) * dimy + j] + r * rotatedVS[(i + dest + 1) * dimy + j];		// WARNING: ACCUMULATED
}


GpuInterface::GpuInterface() {}


// KERNEL-2.2 FROM 46 TO 90 DEGREES
__global__
void rotatedVSComputation45to90(float *sDEM, float *rotatedVS, int angle, int dimy, int dimx, int startGPUbatch, int endGPUbatch, float POVh) {

    int i = blockIdx.x * blockDim.x + threadIdx.x + startGPUbatch;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    float r = (1.0 / cos((90 - angle) * M_PI / 180.0)) * (1.0 / cos((90 - angle) * M_PI / 180.0));

    float tanAngComp = tanf((90 - angle) * M_PI / 180);
    float tanAng = tanf(angle * M_PI / 180);
    int offset = dimy - (dimx - 1) * tanAngComp;
    int threshold = offset + dimy;
    int filafinal = 2 * dimy;

    // if (i == 0 && j == 0) printf("off:%d - th:%d\n", offset, threshold);

    int start, end;
    float cv, h, max_angle, open_delta_d, delta_d, cAngle;
    bool visible, above, opening, closing;

    if (i < endGPUbatch && j < dimx) {

        start = 0;
        end = 0;

        if (i >= offset && i < dimy) {
            start = (dimx - 1) - (i - offset) * tanAng;
            end = dimx;
        }
        else if (i >= dimy && i < threshold) {
            start = 0;
            end = dimx;
        }
        else if (i >= threshold && i < filafinal) {
            start = 0;
            end = dimx - (i - dimy - offset) * tanAng;
        }

        cv = 0;                                                                                 // Ya tengo observador en j
        h = sDEM[i * dimy + j] + POVh;
        max_angle = -2000;
        visible = true;
        open_delta_d = 0;

        for (int k = j - 1; k >= start && k < end; k--) {                                      // Calculo cv a izqda

            delta_d = j - k;
            cAngle = (sDEM[i * dimy + k] - h) / delta_d;
            above = (cAngle > max_angle);
            opening = above && (!visible);
            closing = (!above) && (visible);

            visible = above;
            max_angle = max(cAngle, max_angle);
            if (opening) open_delta_d = delta_d;
            if (closing) cv += (delta_d * delta_d - open_delta_d * open_delta_d);
        }

        max_angle = -2000;
        visible = true;
        open_delta_d = 0;

        for (int k = j + 1; k >= start && k < end; k++) {                                       // Calculo cv a dcha

            delta_d = k - j;
            cAngle = (sDEM[i * dimy + k] - h) / delta_d;
            above = (cAngle > max_angle);
            opening = above && (!visible);
            closing = (!above) && (visible);

            visible = above;
            max_angle = max(cAngle, max_angle);
            if (opening) open_delta_d = delta_d;
            if (closing) cv += (delta_d * delta_d - open_delta_d * open_delta_d);
        }

        rotatedVS[i * dimy + j] = cv * r;
    }

}


GpuInterface::GpuInterface(int dimy, int dimx) {

    this->dimy = dimy;
    this->dimx = dimx;
}

GpuInterface::GpuInterface(int dimy, int dimx, int deviceIndex) {

    this->dimy = dimy;
    this->dimx = dimx;
    this->devIndex = deviceIndex;

    // Select GPU
    cudaError err = cudaSetDevice(deviceIndex);
    if (cudaSuccess != err) {
        std::cout << "Error in selecting device" << std::endl << std::endl;
        #ifdef _WIN32
        WINPAUSE;
        #endif	
        exit( -1 );
    }

    // Fist time cuda API is called (reduce calls time further on)
    int *h_dummy;
    int *d_dummy;
    CudaSafeCall( cudaHostAlloc(&h_dummy, sizeof(*h_dummy), cudaHostAllocPortable) );
    CudaSafeCall( cudaMalloc(&d_dummy, sizeof(*d_dummy)) );
    CudaSafeCall( cudaMemcpy(d_dummy, h_dummy, sizeof(*d_dummy), cudaMemcpyHostToDevice) );
    CudaSafeCall( cudaFree(d_dummy) );
    CudaSafeCall( cudaFreeHost(h_dummy) );
}


void GpuInterface::DeviceProperties() {
     
    std::cout << std::endl << "=========" << std::endl << "CUDA version: v" << CUDART_VERSION << std::endl;    
    
    int devCount;
    int numberOfSMs;
    cudaGetDeviceCount(&devCount);
    std::cout << "CUDA Devices: " << std::endl << std::endl;
    
    for(int i = 0; i < devCount; ++i)
    {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, i);
            cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, i);
            std::cout << i << ": " << props.name << ": " << props.major << "." << props.minor << std::endl;
            std::cout << "  Number of SMs:              " << numberOfSMs << std::endl;
            std::cout << "  Global memory:              " << props.totalGlobalMem / mb << "MB" << std::endl;
            std::cout << "  Shared memory per block:    " << props.sharedMemPerBlock / kb << "kB" << std::endl;
            std::cout << "  Constant memory:            " << props.totalConstMem / kb << "kB" << std::endl;
            std::cout << "  Registers per block:        " << props.regsPerBlock << std::endl;
    
            std::cout << "  Warp size:                  " << props.warpSize << std::endl;
            std::cout << "  Max threads per block:      " << props.maxThreadsPerBlock << std::endl;
            std::cout << "  Max block dimensions:       [" << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << "]" << std::endl;
            std::cout << "  Max grid dimensions:        [" << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << "]" << std::endl;
            std::cout << std::endl;
    }

    std::cout << "=========" << std::endl << std::endl;
}


CpuInterface::CpuInterface() {}


CpuInterface::CpuInterface(int dimy, int dimx) {

    this->dimy = dimy;
    this->dimx = dimx;
}


void CpuInterface::AllocDEMHost(double *&heights, float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float **&h_multiSector, int devCount, float *&h_totalVS) {

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

void CpuInterface::AllocDEMHost(double *&heights, float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float *&h_totalVS) {

    CudaSafeCall( cudaHostAlloc(&h_DEM, dimy * dimx * sizeof(*h_DEM), cudaHostAllocPortable) );
    CudaSafeCall( cudaHostAlloc(&h_sDEM, 2 * dimy * dimx * sizeof(*h_sDEM), cudaHostAllocPortable) );

    CudaSafeCall( cudaHostAlloc(&h_rotatedVS, 2 * dimy * dimx * sizeof(*h_rotatedVS), cudaHostAllocPortable) );
    CudaSafeCall( cudaHostAlloc(&h_sectorVS, dimy * dimx * sizeof(*h_sectorVS), cudaHostAllocPortable) );

    CudaSafeCall( cudaHostAlloc(&h_totalVS, dimy * dimx * sizeof(*h_totalVS), cudaHostAllocPortable) );

    #ifdef DEBUG
    size = dimy * dimx * (sizeof(*h_DEM) + 2 * sizeof(*h_sDEM) + 2 * sizeof(*h_rotatedVS) + sizeof(*h_sectorVS) + sizeof(*h_totalVS));
    std::cout << "Total memory allocated in host: " << size / mb << " Mb" << std::endl;
    #endif
}


void GpuInterface::AllocDEMDevice(int deviceIndex) {

    cudaSetDevice(deviceIndex);
    CudaSafeCall( cudaMalloc(&d_DEM, dimy * dimx * sizeof(*d_DEM)) );
    CudaSafeCall( cudaMalloc(&d_sDEM, 2 * dimy * dimx * sizeof(*d_sDEM)) );
    CudaSafeCall( cudaMemset(d_sDEM, 0, 2 * dimy * dimx * sizeof(*d_sDEM)) );

    CudaSafeCall( cudaMalloc(&d_rotatedVS, 2 * dimy * dimx * sizeof(*d_rotatedVS)) );
    CudaSafeCall( cudaMalloc(&d_sectorVS, dimy * dimx * sizeof(*d_sectorVS)) );
    CudaSafeCall( cudaMalloc(&d_totalVS, dimy * dimx * sizeof(*d_totalVS)) );

    #ifdef DEBUG
    size = dimy * dimx * (sizeof(*d_DEM) + 2 * sizeof(*d_sDEM) + 2 * sizeof(*d_rotatedVS) + sizeof(*d_sectorVS) + sizeof(*d_totalVS));
    std::cout << "Total memory allocated in device: " << size / mb << " Mb" << std::endl;
    #endif
}


void GpuInterface::MemcpyDEM_H2D(float *&h_DEM, int deviceIndex) {

    cudaSetDevice(deviceIndex);
    CudaSafeCall( cudaMemcpy(d_DEM, h_DEM, dimy * dimx * sizeof(*d_DEM), cudaMemcpyHostToDevice) );
}


void GpuInterface::MemcpyDEM_D2H(float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float *&h_totalVS) {

    // CudaSafeCall( cudaMemcpy(h_sDEM, d_sDEM, 2 * dimy * dimx * sizeof(*h_sDEM), cudaMemcpyDeviceToHost) );
    // CudaSafeCall( cudaMemcpy(h_rotatedVS, d_rotatedVS, 2 * dimy * dimx * sizeof(*h_rotatedVS), cudaMemcpyDeviceToHost) );
    CudaSafeCall( cudaMemcpy(h_sectorVS, d_sectorVS, dimy * dimx * sizeof(*h_sectorVS), cudaMemcpyDeviceToHost) );
    
    // CudaSafeCall( cudaMemcpy(h_totalVS, d_totalVS, dimy * dimx * sizeof(*h_totalVS), cudaMemcpyDeviceToHost) );
}


void GpuInterface::MemcpyDEM_D2Hheterogeneous(float *&h_rotatedVS, int startGPUbatch, int endGPUbatch, int deviceIndex) {

    int gpu_id = -1;
    cudaSetDevice(deviceIndex);
    cudaGetDevice(&gpu_id);
    printf("CPU thread %d uses CUDA device memcpy %d\n", deviceIndex, gpu_id);
    CudaSafeCall( cudaMemcpyAsync(&h_rotatedVS[startGPUbatch * dimx], &d_rotatedVS[startGPUbatch * dimx], (endGPUbatch - startGPUbatch) * dimx * sizeof(*h_rotatedVS), cudaMemcpyDeviceToHost, 0) );
}


void GpuInterface::Syncronize(int deviceIndex) {

    int gpu_id = -1;
    cudaSetDevice(deviceIndex);
    cudaDeviceSynchronize();
    cudaGetDevice(&gpu_id);
    // printf("CPU thread %d uses CUDA device synchronization %d\n", deviceIndex, gpu_id);
}


void GpuInterface::GetNumberGPUs(int &devCount) {

    cudaGetDeviceCount(&devCount);
}



void GpuInterface::Execute(float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float *&h_totalVS, int angle, int startGPUbatch, int endGPUbatch, int deviceIndex, float POVh) {

    AllocDEMDevice(deviceIndex);

    MemcpyDEM_H2D(h_DEM, deviceIndex);

    #ifdef DEBUG
    std::cout << std::endl << "Obtaining skewed DEM on GPU" << std::endl;
    startGPU = std::chrono::high_resolution_clock::now();
    #endif
    dim3 threadsPerBlock(8, 8);
    int gx = (dimy % threadsPerBlock.x == 0) ? dimy / threadsPerBlock.x : dimy / threadsPerBlock.x + 1;
    int gy = (dimx % threadsPerBlock.y == 0) ? dimx / threadsPerBlock.y : dimx / threadsPerBlock.y + 1;    
    dim3 blocksPerGrid(gx, gy);
    cudaSetDevice(deviceIndex);
    optimizedRotation0to45<<< blocksPerGrid, threadsPerBlock >>>(d_sDEM, d_DEM, angle, dimy, dimx);
    #ifdef DEBUG
    CudaCheckError();
    endGPU = std::chrono::high_resolution_clock::now();
    std::cout << "sDEM Calculation kernel (ms): " << (double)(endGPU - startGPU).count() / 1000000 << std::endl;
    #endif

    #ifdef DEBUG
    std::cout << std::endl << "Obtaining sector viewshed on skewed map on GPU" << std::endl;
    startGPU = std::chrono::high_resolution_clock::now();
    #endif
    int chunkSize = endGPUbatch - startGPUbatch;
    blocksPerGrid.x = (chunkSize % threadsPerBlock.x == 0) ? chunkSize / threadsPerBlock.x : chunkSize / threadsPerBlock.x + 1;
    cudaSetDevice(deviceIndex);
    rotatedVSComputation0to45<<< blocksPerGrid, threadsPerBlock >>>(d_sDEM, d_rotatedVS, angle, dimy, dimx, startGPUbatch, endGPUbatch, POVh);
    #ifdef DEBUG
    CudaCheckError();
    endGPU = std::chrono::high_resolution_clock::now();
    std::cout << "sDEM Viewshed Calculation kernel (ms): " << (double)(endGPU - startGPU).count() / 1000000 << std::endl;
    #endif
}


void GpuInterface::CalculateVS(int angle) {

    dim3 threadsPerBlock(8, 8);
    int gx = (dimy % threadsPerBlock.x == 0) ? dimy / threadsPerBlock.x : dimy / threadsPerBlock.x + 1;
    int gy = (dimx % threadsPerBlock.y == 0) ? dimx / threadsPerBlock.y : dimx / threadsPerBlock.y + 1;    
    dim3 blocksPerGrid(gx, gy);

    #ifdef DEBUG
    std::cout << std::endl << "Obtaining sector viewshed on original DEM on GPU" << std::endl;
    startGPU = std::chrono::high_resolution_clock::now();
    #endif
    viewshedCalculation0to45<<< blocksPerGrid, threadsPerBlock >>>(d_sectorVS, d_rotatedVS, angle, dimy, dimx);
    #ifdef DEBUG
    CudaCheckError();
    endGPU = std::chrono::high_resolution_clock::now();
    std::cout << "DEM Viewshed Calculation kernel (ms): " << (double)(endGPU - startGPU).count() / 1000000 << std::endl;
    #endif
}


void GpuInterface::ExecuteSingleSectorMultiGPUmaster(float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float *&h_totalVS, int angle, int devCount, float POVh) {
            
    float **DEM = new float*[devCount];
    float **sDEM = new float*[devCount];
    float **rotatedVS = new float*[devCount];

    // Streams creation
    cudaStream_t *stream = new cudaStream_t[devCount];

    // Ranges of balanced execution for each GPU
    int sch_s38[4][5] = { {0, 2 * dimy, 0, 0, 0},
                      {0, dimy + (dimy / 9), 2 * dimy, 0, 0},
                      {0, (int)(0.492 * 2 * dimy), (int)(0.615 * 2 * dimy), 2 * dimy, 0},
                      {0, (int)(0.458 * 2 * dimy), (int)(0.55 * 2 * dimy), (int)(0.648 * 2 * dimy), 2 * dimy} }; 

    // Create streams for issuing GPU command asynchronously and allocate memory (GPU and System page-locked)
    for (int i = 0; i < devCount; i++)
    {       
        // Select GPU
        cudaError err = cudaSetDevice(i);
        if (cudaSuccess != err) {
            std::cout << "Error in selecting device" << std::endl << std::endl;
            #ifdef _WIN32
            WINPAUSE;
            #endif	
            exit(-1);
        }

        CudaSafeCall( cudaStreamCreate(&stream[i]) );

        // Set and check the CUDA device for this CPU thread
        int gpu_id = -1;
        cudaGetDevice(&gpu_id);
        printf("Iteration %d launches CUDA device %d\n", i, gpu_id);

        // Memory allocation
        CudaSafeCall( cudaMalloc(&DEM[i], dimy * dimx * sizeof(*d_DEM)) );
        CudaSafeCall( cudaMalloc(&sDEM[i], 2 * dimy * dimx * sizeof(*d_sDEM)) );
        CudaSafeCall( cudaMemset(sDEM[i], 0, 2 * dimy * dimx * sizeof(*d_sDEM)) );
        CudaSafeCall( cudaMalloc(&rotatedVS[i], 2 * dimy * dimx * sizeof(*d_rotatedVS)) );
        CudaSafeCall( cudaMemset(rotatedVS[i], 0, 2 * dimy * dimx * sizeof(*d_sDEM)) );
        // CudaSafeCall( skewCudaMalloc(&d_sectorVS, dimy * dimx * sizeof(*d_sectorVS)) );
        // CudaSafeCall( skewCudaMalloc(&d_totalVS, dimy * dimx * sizeof(*d_totalVS)) );
    }

    for (int i = 0; i < devCount; i++)
    {        
        // Select GPU
        cudaError err = cudaSetDevice(i);
        if (cudaSuccess != err) {
            std::cout << "Error in selecting device" << std::endl << std::endl;
            #ifdef _WIN32
            WINPAUSE;
            #endif	
            exit(-1);
        }

        // H2D
        CudaSafeCall( cudaMemcpyAsync(DEM[i], h_DEM, dimy * dimx * sizeof(*d_DEM), cudaMemcpyHostToDevice, stream[i]) );
	}
	
	for (int i = 0; i < devCount; i++)
    {
		// int startGPUbatch = 2 * dimy * i / devCount;
        // int endGPUbatch = 2 * dimy * (i + 1) / devCount;
        int startGPUbatch = sch_s38[devCount - 1][i];
        int endGPUbatch = sch_s38[devCount - 1][i + 1];
        int batchSize = endGPUbatch - startGPUbatch;
        
        // Select GPU
        cudaError err = cudaSetDevice(i);
        if (cudaSuccess != err) {
            std::cout << "Error in selecting device" << std::endl << std::endl;
            #ifdef _WIN32
            WINPAUSE;
            #endif	
            exit(-1);
        }

        // Kernel-1
        #ifdef DEBUG
        std::cout << std::endl << "Obtaining skewed DEM on GPU" << std::endl;
        startGPU = std::chrono::high_resolution_clock::now();
        #endif
        dim3 threadsPerBlock(8, 8);
        int gx = (dimy % threadsPerBlock.x == 0) ? dimy / threadsPerBlock.x : dimy / threadsPerBlock.x + 1;
        int gy = (dimx % threadsPerBlock.y == 0) ? dimx / threadsPerBlock.y : dimx / threadsPerBlock.y + 1;    
        dim3 blocksPerGrid(gx, gy);
        optimizedRotation0to45<<< blocksPerGrid, threadsPerBlock, 0, stream[i] >>>(sDEM[i], DEM[i], angle, dimy, dimx);
        #ifdef DEBUG
        CudaCheckError();
        endGPU = std::chrono::high_resolution_clock::now();
        std::cout << "sDEM Calculation kernel (ms): " << (double)(endGPU - startGPU).count() / 1000000 << std::endl;
        #endif

        // Kernel-2
        #ifdef DEBUG
        std::cout << std::endl << "Obtaining sector viewshed on skewed map on GPU" << std::endl;
        startGPU = std::chrono::high_resolution_clock::now();
        #endif
        blocksPerGrid.x = (batchSize % threadsPerBlock.x == 0) ? batchSize / threadsPerBlock.x : batchSize / threadsPerBlock.x + 1;
        rotatedVSComputation0to45<<< blocksPerGrid, threadsPerBlock, 0, stream[i] >>>(sDEM[i], rotatedVS[i], angle, dimy, dimx, startGPUbatch, endGPUbatch, POVh);
        #ifdef DEBUG
        CudaCheckError();
        endGPU = std::chrono::high_resolution_clock::now();
        std::cout << "sDEM Viewshed Calculation kernel (ms): " << (double)(endGPU - startGPU).count() / 1000000 << std::endl;
        #endif

        // D2H
        CudaSafeCall( cudaMemcpyAsync(&h_rotatedVS[startGPUbatch * dimx], &rotatedVS[i][startGPUbatch * dimx], batchSize * dimx * sizeof(*h_rotatedVS), cudaMemcpyDeviceToHost, stream[i]) );
    }
           
    for (int i = 0; i < devCount; i++)
    {    
        // Select GPU
        cudaError err = cudaSetDevice(i);
        if (cudaSuccess != err) {
            std::cout << "Error in selecting device" << std::endl << std::endl;
            #ifdef _WIN32
            WINPAUSE;
            #endif	
            exit(-1);
        }
        
        // Wait for all operations to finish
        cudaStreamSynchronize(stream[i]);
        
        CudaSafeCall( cudaFree(DEM[i]) );
		CudaSafeCall( cudaFree(sDEM[i]) );
		CudaSafeCall( cudaFree(rotatedVS[i]) );
		// CudaSafeCall( cudaFree(d_sectorVS) );
		// CudaSafeCall( cudaFree(d_totalVS) );
        
        cudaStreamDestroy(stream[i]);
    }
    
    delete[] DEM, sDEM, rotatedVS;
    delete stream;
}


void GpuInterface::ExecuteAccMultiGPUmaster(float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float **&h_multiSectorVS, float *&h_totalVS, int maxAngle, int devCount, float POVh) {

    // std::cout << std::endl << "Number of GPUs: " << devCount << std::endl;
            
    float **DEM = new float*[devCount];
    float **sDEM = new float*[devCount];
    float **rotatedVS = new float*[devCount];
    float **sectorVS = new float*[devCount];

    // STREAMS CREATION
    cudaStream_t *stream = new cudaStream_t[devCount];

    //Create streams for issuing GPU command asynchronously and allocate memory (GPU and System page-locked)
    for (int i = 0; i < devCount; i++)
    {       
        // Select GPU
        cudaError err = cudaSetDevice(i);
        if (cudaSuccess != err) {
            std::cout << "Error in selecting device" << std::endl << std::endl;
            #ifdef _WIN32
            WINPAUSE;
            #endif	
            exit(-1);
        }

        CudaSafeCall( cudaStreamCreate(&stream[i]) );

        // set and check the CUDA device for this CPU thread
        // int gpu_id = -1;
        // cudaGetDevice(&gpu_id);
        // printf("Iteration %d launches CUDA device %d\n", i, gpu_id);

        // Memory allocation
        CudaSafeCall( cudaMalloc(&DEM[i], dimy * dimx * sizeof(*d_DEM)) );
        CudaSafeCall( cudaMalloc(&sDEM[i], 2 * dimy * dimx * sizeof(*d_sDEM)) );
        // CudaSafeCall( cudaMemset(sDEM[i], 0, 2 * dimy * dimx * sizeof(*d_sDEM)) );
        CudaSafeCall( cudaMalloc(&rotatedVS[i], 2 * dimy * dimx * sizeof(*d_rotatedVS)) );
        // CudaSafeCall( cudaMemset(rotatedVS[i], 0, 2 * dimy * dimx * sizeof(*d_sDEM)) );
        CudaSafeCall( cudaMalloc(&sectorVS[i], dimy * dimx * sizeof(*d_sectorVS)) );
        CudaSafeCall( cudaMemsetAsync(sectorVS[i], 0, dimy * dimx * sizeof(*d_sDEM), stream[i]) );
        // CudaSafeCall( skewCudaMalloc(&d_totalVS, dimy * dimx * sizeof(*d_totalVS)) );
    }

    for (int i = 0; i < devCount; i++)
    {        
        // Select GPU
        cudaError err = cudaSetDevice(i);
        if (cudaSuccess != err) {
            std::cout << "Error in selecting device" << std::endl << std::endl;
            #ifdef _WIN32
            WINPAUSE;
            #endif	
            exit(-1);
        }

        // H2D
        CudaSafeCall( cudaMemcpyAsync(DEM[i], h_DEM, dimy * dimx * sizeof(*d_DEM), cudaMemcpyHostToDevice, stream[i]) );
	}
    
    int currGPU;
    for (int ang = 0; ang < maxAngle; ang++)
    {
        currGPU = ang % devCount;
        
        CudaSafeCall( cudaMemsetAsync(sDEM[currGPU], 0, 2 * dimy * dimx * sizeof(*d_sDEM), stream[currGPU]) );
        CudaSafeCall( cudaMemsetAsync(rotatedVS[currGPU], 0, 2 * dimy * dimx * sizeof(*d_sDEM), stream[currGPU]) );

        // Select GPU
        cudaError err = cudaSetDevice(currGPU);
        if (cudaSuccess != err) {
            std::cout << "Error in selecting device" << std::endl << std::endl;
            #ifdef _WIN32
            WINPAUSE;
            #endif	
            exit(-1);
        }

        int startGPUbatch = 0;
        int endGPUbatch = 2 * dimy;
        int batchSize = endGPUbatch - startGPUbatch;

        // Kernel-1
        #ifdef DEBUG
        std::cout << std::endl << "Obtaining skewed DEM on GPU" << std::endl;
        startGPU = std::chrono::high_resolution_clock::now();
        #endif
        dim3 threadsPerBlock(8, 8);
        int gx = (dimy % threadsPerBlock.x == 0) ? dimy / threadsPerBlock.x : dimy / threadsPerBlock.x + 1;
        int gy = (dimx % threadsPerBlock.y == 0) ? dimx / threadsPerBlock.y : dimx / threadsPerBlock.y + 1;    
        dim3 blocksPerGrid(gx, gy);
        if (ang <= 45) optimizedRotation0to45<<< blocksPerGrid, threadsPerBlock, 0, stream[currGPU] >>>(sDEM[currGPU], DEM[currGPU], ang, dimy, dimx);
        else if (ang > 45 && ang <= 90) optimizedRotation45to90<<< blocksPerGrid, threadsPerBlock, 0, stream[currGPU] >>>(sDEM[currGPU], DEM[currGPU], ang, dimy, dimx);
        else if (ang > 90 && ang <= 135) optimizedRotation90to135<<< blocksPerGrid, threadsPerBlock, 0, stream[currGPU] >>>(sDEM[currGPU], DEM[currGPU], ang, dimy, dimx);
        else if (ang > 135 && ang < 180) optimizedRotation135to180<<< blocksPerGrid, threadsPerBlock, 0, stream[currGPU] >>>(sDEM[currGPU], DEM[currGPU], ang, dimy, dimx);
        #ifdef DEBUG
        CudaCheckError();
        endGPU = std::chrono::high_resolution_clock::now();
        std::cout << "sDEM Calculation kernel (ms): " << (double)(endGPU - startGPU).count() / 1000000 << std::endl;
        #endif

        // Kernel-2
        #ifdef DEBUG
        std::cout << std::endl << "Obtaining sector viewshed on skewed map on GPU" << std::endl;
        startGPU = std::chrono::high_resolution_clock::now();
        #endif
        blocksPerGrid.x = (batchSize % threadsPerBlock.x == 0) ? batchSize / threadsPerBlock.x : batchSize / threadsPerBlock.x + 1;
        if (ang <= 45) rotatedVSComputation0to45<<< blocksPerGrid, threadsPerBlock, 0, stream[currGPU] >>>(sDEM[currGPU], rotatedVS[currGPU], ang, dimy, dimx, startGPUbatch, endGPUbatch, POVh);
        else if (ang > 45 && ang <= 90) rotatedVSComputation45to90<<< blocksPerGrid, threadsPerBlock, 0, stream[currGPU] >>>(sDEM[currGPU], rotatedVS[currGPU], ang, dimy, dimx, startGPUbatch, endGPUbatch, POVh);
        else if (ang > 90 && ang <= 135) rotatedVSComputation90to135<<< blocksPerGrid, threadsPerBlock, 0, stream[currGPU] >>>(sDEM[currGPU], rotatedVS[currGPU], ang, dimy, dimx, startGPUbatch, endGPUbatch, POVh);
        else if (ang > 135 && ang < 180) rotatedVSComputation135to180<<< blocksPerGrid, threadsPerBlock, 0, stream[currGPU] >>>(sDEM[currGPU], rotatedVS[currGPU], ang, dimy, dimx, startGPUbatch, endGPUbatch, POVh);
        #ifdef DEBUG
        CudaCheckError();
        endGPU = std::chrono::high_resolution_clock::now();
        std::cout << "sDEM Viewshed Calculation kernel (ms): " << (double)(endGPU - startGPU).count() / 1000000 << std::endl;
        #endif

        // Kernel-3
        #ifdef DEBUG
        std::cout << std::endl << "Obtaining sector viewshed on original DEM on GPU" << std::endl;
        startGPU = std::chrono::high_resolution_clock::now();
        #endif
        blocksPerGrid.x = (dimy % threadsPerBlock.x == 0) ? dimy / threadsPerBlock.x : dimy / threadsPerBlock.x + 1;
        if (ang <= 45) viewshedCalculation0to45<<< blocksPerGrid, threadsPerBlock >>>(sectorVS[currGPU], rotatedVS[currGPU], ang, dimy, dimx);
        else if (ang > 45 && ang <= 90) viewshedCalculation45to90<<< blocksPerGrid, threadsPerBlock >>>(sectorVS[currGPU], rotatedVS[currGPU], ang, dimy, dimx);
        else if (ang > 90 && ang <= 135) viewshedCalculation90to135<<< blocksPerGrid, threadsPerBlock >>>(sectorVS[currGPU], rotatedVS[currGPU], ang, dimy, dimx);
        else if (ang > 135 && ang < 180) viewshedCalculation135to180<<< blocksPerGrid, threadsPerBlock >>>(sectorVS[currGPU], rotatedVS[currGPU], ang, dimy, dimx);
        #ifdef DEBUG
        CudaCheckError();
        endGPU = std::chrono::high_resolution_clock::now();
        std::cout << "DEM Viewshed Calculation kernel (ms): " << (double)(endGPU - startGPU).count() / 1000000 << std::endl;
        #endif

        // D2H
        // CudaSafeCall( cudaMemcpyAsync(&h_rotatedVS[0], &rotatedVS[currGPU][0], 2 * dimy * dimx * sizeof(*h_rotatedVS), cudaMemcpyDeviceToHost, stream[currGPU]) );
        if (maxAngle - ang <= devCount) CudaSafeCall( cudaMemcpyAsync(&h_multiSectorVS[currGPU][0], &sectorVS[currGPU][0], dimy * dimx * sizeof(float), cudaMemcpyDeviceToHost, stream[currGPU]) );
    }
           
    for (int i = 0; i < devCount; i++)
    {    
        // Select GPU
        cudaError err = cudaSetDevice(i);
        if (cudaSuccess != err) {
            std::cout << "Error in selecting device" << std::endl << std::endl;
            exit(-1);
        }
        
        // Wait for all operations to finish
        cudaStreamSynchronize(stream[i]);
        
        CudaSafeCall( cudaFree(DEM[i]) );
		CudaSafeCall( cudaFree(sDEM[i]) );
		CudaSafeCall( cudaFree(rotatedVS[i]) );
		CudaSafeCall( cudaFree(sectorVS[i]) );
		// CudaSafeCall( cudaFree(d_totalVS) );
        
        cudaStreamDestroy(stream[i]);
    }
    
    delete[] DEM, sDEM, rotatedVS, sectorVS;
    delete stream;
}


void CpuInterface::FreeHostMemory(float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float *&h_totalVS) {

    CudaSafeCall( cudaFreeHost(h_DEM) );
    CudaSafeCall( cudaFreeHost(h_sDEM) );
    CudaSafeCall( cudaFreeHost(h_rotatedVS) );
    CudaSafeCall( cudaFreeHost(h_sectorVS) );
    
    CudaSafeCall( cudaFreeHost(h_totalVS) );
}


void CpuInterface::FreeHostMemory(float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float **&h_multiSectorVS, int devCount, float *&h_totalVS) {

    CudaSafeCall( cudaFreeHost(h_DEM) );
    CudaSafeCall( cudaFreeHost(h_sDEM) );
    CudaSafeCall( cudaFreeHost(h_rotatedVS) );
    CudaSafeCall( cudaFreeHost(h_sectorVS) );
    
    for (int i = 0; i < devCount; i++)
		CudaSafeCall( cudaFreeHost(h_multiSectorVS[i]) );
	CudaSafeCall( cudaFreeHost(h_multiSectorVS) );
    
    CudaSafeCall( cudaFreeHost(h_totalVS) );
}


void GpuInterface::FreeDeviceMemory() {

    CudaSafeCall( cudaFree(d_DEM) );
    CudaSafeCall( cudaFree(d_sDEM) );

    CudaSafeCall( cudaFree(d_rotatedVS) );
    CudaSafeCall( cudaFree(d_sectorVS) );
    CudaSafeCall( cudaFree(d_totalVS) );
}


// // NEED TO BE FIXED THE SAME WAY THE MASTER WAS
// void GpuInterface::ExecuteMultiGPUopenmp(float *&h_DEM, float *&h_sDEM, float *&h_rotatedVS, float *&h_sectorVS, float *&h_totalVS, int angle) {

//     int devCount = 1;
//     // cudaGetDeviceCount(&devCount);

//     #pragma omp parallel num_threads(devCount)
//     {
//         int tid = omp_get_thread_num();
        
//         int startGPUbatch = 2 * dimy * tid / devCount;
//         int endGPUbatch = 2 * dimy * (tid + 1) / devCount;
//         int batchSize = endGPUbatch - startGPUbatch;
        
//         // Select GPU
//         cudaError err = cudaSetDevice(tid);
//         if (cudaSuccess != err) {
//             std::cout << "Error in selecting device" << std::endl << std::endl;
//             #ifdef _WIN32
//             WINPAUSE;
//             #endif	
//             exit(-1);
//         }
        
//         // set and check the CUDA device for this CPU thread
//         int gpu_id = -1;
//         cudaGetDevice(&gpu_id);
//         printf("CPU thread %d uses CUDA device %d\n", tid, gpu_id);

//         // Memory allocation
//         CudaSafeCall( skewCudaMalloc(&d_DEM, dimy * dimx * sizeof(*d_DEM)) );
//         CudaSafeCall( skewCudaMalloc(&d_sDEM, 2 * dimy * dimx * sizeof(*d_sDEM)) );
//         CudaSafeCall( cudaMemset(d_sDEM, 0, 2 * dimy * dimx * sizeof(*d_sDEM)) );
//         CudaSafeCall( skewCudaMalloc(&d_rotatedVS, 2 * dimy * dimx * sizeof(*d_rotatedVS)) );
//         CudaSafeCall( cudaMemset(d_rotatedVS, 0, 2 * dimy * dimx * sizeof(*d_sDEM)) );
//         // CudaSafeCall( skewCudaMalloc(&d_sectorVS, dimy * dimx * sizeof(*d_sectorVS)) );
//         // CudaSafeCall( skewCudaMalloc(&d_totalVS, dimy * dimx * sizeof(*d_totalVS)) );

//         // H2D
//         CudaSafeCall( cudaMemcpy(d_DEM, h_DEM, dimy * dimx * sizeof(*d_DEM), cudaMemcpyHostToDevice) );

//         // Kernel-1
//         #ifdef DEBUG
//         std::cout << std::endl << "Obtaining skewed DEM on GPU" << std::endl;
//         startGPU = std::chrono::high_resolution_clock::now();
//         #endif
//         dim3 threadsPerBlock(8, 8);
//         int gx = (dimy % threadsPerBlock.x == 0) ? dimy / threadsPerBlock.x : dimy / threadsPerBlock.x + 1;
//         int gy = (dimx % threadsPerBlock.y == 0) ? dimx / threadsPerBlock.y : dimx / threadsPerBlock.y + 1;    
//         dim3 blocksPerGrid(gx, gy);
//         optimizedRotation0to45<<< blocksPerGrid, threadsPerBlock >>>(d_sDEM, d_DEM, angle, dimy, dimx);
//         #ifdef DEBUG
//         CudaCheckError();
//         endGPU = std::chrono::high_resolution_clock::now();
//         std::cout << "sDEM Calculation kernel (ms): " << (double)(endGPU - startGPU).count() / 1000000 << std::endl;
//         #endif

//         // Kernel-2
//         #ifdef DEBUG
//         std::cout << std::endl << "Obtaining sector viewshed on skewed map on GPU" << std::endl;
//         startGPU = std::chrono::high_resolution_clock::now();
//         #endif
//         blocksPerGrid.x = (batchSize % threadsPerBlock.x == 0) ? batchSize / threadsPerBlock.x : batchSize / threadsPerBlock.x + 1;
//         rotatedVSComputation0to45<<< blocksPerGrid, threadsPerBlock >>>(d_sDEM, d_rotatedVS, angle, dimy, dimx, startGPUbatch, endGPUbatch);
//         #ifdef DEBUG
//         CudaCheckError();
//         endGPU = std::chrono::high_resolution_clock::now();
//         std::cout << "sDEM Viewshed Calculation kernel (ms): " << (double)(endGPU - startGPU).count() / 1000000 << std::endl;
//         #endif

//         // D2H
//         CudaSafeCall( cudaMemcpy(&h_rotatedVS[startGPUbatch * dimx], &d_rotatedVS[startGPUbatch * dimx], batchSize * dimx * sizeof(*h_rotatedVS), cudaMemcpyDeviceToHost) );

//         // Synchronization
//         cudaDeviceSynchronize();
        
//         // Free
//         CudaSafeCall( cudaFree(d_DEM) );
// 		CudaSafeCall( cudaFree(d_sDEM) );
// 		CudaSafeCall( cudaFree(d_rotatedVS) );
// 		// CudaSafeCall( cudaFree(d_sectorVS) );
// 		// CudaSafeCall( cudaFree(d_totalVS) );
//     }
    
// }
