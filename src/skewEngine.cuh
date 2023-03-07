//
// Created by Felipe on 23-11-22.
//
// Clase para mapear datos 2D de entrada en datos con sesgo (data -> skew)
//
// Los datos en formato sesgado tienen alineados datos en memoria que en
// formato original no lo estarían
//
// Tras el procesamiento (por una función externa a la clase),
// Los datos procesados se remapean a la posición original (skeOut -> Out)
//

//

#ifndef SKEWENGINE_H
#define SKEWENGINE_H
#include <cmath>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>

#define _USE_MATH_DEFINES
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif



// ToDo error checks

#ifdef _WIN32
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#else
#endif
#define cudaHostAllocPortable 0x01



#if defined(__CUDACC__)


template <typename T>
__global__
void cudaSkew(T *skewed, T *unskewed, double skewness, int dim_o, int dim_i) {

    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    float y = skewness * col;
    int dest = y;
    float r = y - dest;

    if (row < dim_o && col < dim_i) {
        atomicAdd(&skewed[(row + dest) * dim_i + col ], (1.0 - r) * unskewed[dim_i * row + col]);
        atomicAdd(&skewed[(row + dest + 1) * dim_i + col ], r * unskewed[dim_i * row + col]);
    }
}

template <typename T>
__global__
void cudaDeskew(T *unskewed, T *skewed, double skewness, int dim_o, int dim_i) {

    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    float y = skewness * col;
    int dest = y;
    float r = y - dest;

    if (row < dim_o && col < dim_i)
        unskewed[row * dim_i + col] +=
                (1.0 - r) * skewed[(row + dest) * dim_i + col]
                      + r * skewed[(row + dest + 1) * dim_i + col];
}

template <typename T>
__global__
void cudaSkewV0(T *skewed, T *unskewed, double skewness, int dim_o, int dim_i, float *w, int *t) {

    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    //float y = skewness * col;
    //int dest = y;
    //float r = y - dest;

    if (row < dim_o && col < dim_i) {
        atomicAdd(&skewed[(row + t[col]) * dim_i + col ], (1.0 - w[col]) * unskewed[dim_i * row + col]);
        atomicAdd(&skewed[(row + t[col] + 1) * dim_i + col ], w[col] * unskewed[dim_i * row + col]);
    }
}

template <typename T>
__global__
void cudaDeskewV0(T *unskewed, T *skewed, double skewness, int dim_o, int dim_i, float *w, int *t) {

    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

//    float y = skewness * col;
//    int dest = y;
//    float r = y - dest;

    if (row < dim_o && col < dim_i)
        unskewed[row * dim_i + col] +=
                (1.0 - w[col]) * skewed[(row + t[col]) * dim_i + col]
                + w[col] * skewed[(row + t[col] + 1) * dim_i + col];
}

#endif


template <typename T>
struct inputData{
    T *input0;
    T *input1;
    T *input2;
    T *input3;
};




// T can be float, double, ...
template <typename T>
class skewEngine {
    typedef void (*pkernelfunc)(skewEngine<T> *skewer);


    inputData<T> input;
    int dimx;
    int dimy;
    int N;
    float fAngle;
    double torads=M_PI/180.0;

    // Regular or Transposed
    bool isT;
    // Below or above 90, is mirrored
    bool isM;

    //inData_t input;
    //T *input0;
    //T *input1;
    //T *input2;
    //T *input3;
    T *output0=nullptr;
    T *output1=nullptr;
    T *output2=nullptr;
    T *output3=nullptr;
    int dim_l;
#if defined(__CUDACC__)
    cudaStream_t stream;
#endif

public:

    skewEngine(int dimx, int dimy, inputData<T> inData , bool isGPU=false, int deviceId=0);
    virtual ~skewEngine();
    void computeParams(int angle);
    void skewShape();
    void resetSkew();
    void skew(int angle);
    void deskew(int isMax=0);
    void reduce(T **output);
    void reduce(T *output, int isMax=0);

    static inputData<T> prepare(T *inputD, int dimx, int dimy);
    static inputData<T> prepare( inputData<T> *input, int dimx, int dimy);
    static void allocate(inputData<T> &inData,T* inputD,T** outD,int dimx, int dimy);
    static void deallocate(inputData<T> inData,T* inputD,T* outD);
    void kernel();


    bool useV0=true; //indifferent  (weigth and target precomputed or not in gpu)

    //  *******************************************************************************
    //  Cuda section
    //  *******************************************************************************
    void skewCudaMalloc();
    void skewCudaFree();
    int sectorType;

    bool isGPU;
    int deviceId;
    T *d_input;
    T *d_skewInput;
    T *d_skewOutput;
    T *d_output;
    int *target;
    float *weight;
    int *d_target;
    float *d_weight;
    short unsigned int *d_first;
    short unsigned int *d_last;

    int startGPUbatch ;
    int endGPUbatch;
    int batchSize;
    //  *******************************************************************************


    T *skewInput;
    T *skewOutput;
    int dim_skewx,dim_skewy;
    int skewHeight, skewWidth;
    short unsigned int *first;
    short unsigned int *last;
    int a;
    int newAngle;
    double scale;
    int offset;
    double skewness,iskewness;
    int dim_i,dim_o;

    void (*kernelgpu)(T *d_skewOutput,T *d_skewInput,int dim_i,int skewHeight,unsigned short *d_first,unsigned short *d_last, T val, int angle);
    void (*kernelcpu)(skewEngine<T> *);
};

/*
 * Static
 */
template<typename T>
void skewEngine<T>::deallocate(inputData<T> inData,T* inputD,T* outD) {
    cudaFreeHost(inData.input0) ;
    cudaFreeHost(inData.input1) ;
    cudaFreeHost(inData.input2) ;
    cudaFreeHost(inData.input3) ;
    free(outD);
    free(inputD);
}

template<typename T>
void skewEngine<T>::allocate(inputData<T> &inData,T* inputD,T** outD,int dimx, int dimy)
{

    // AllocDEMHost (in auxf.cu) allocate arrays in CPU in a better way than malloc, if it's going to be
    // used in CUDA  ("pinned" memory)

    //
    int dim=dimx*dimy;
    int dataSize=sizeof(T);

    //cpu->AllocDEMHost(inData.input0,inData.input1,inData.input2,inData.input3,dim);
    cudaHostAlloc(&inData.input0, dim* sizeof(*inData.input0), cudaHostAllocPortable) ;
    cudaHostAlloc(&inData.input1, dim* sizeof(*inData.input1), cudaHostAllocPortable) ;
    cudaHostAlloc(&inData.input2, dim* sizeof(*inData.input2), cudaHostAllocPortable) ;
    cudaHostAlloc(&inData.input3, dim* sizeof(*inData.input3), cudaHostAllocPortable) ;

    memcpy(inData.input0,inputD,dim*dataSize); //Move input data to pinned memory

    *outD=(T *)malloc(dim*sizeof (T));
    memset(*outD,0,dim*dataSize);
    // Cambiar, si no es sdem ni blur
    inData=  skewEngine<float>::prepare(&inData,dimx,dimy);// Rotated and mirror
}










// dimx, dimy, input has been initiated with arguments values
template<typename T>
skewEngine<T>::skewEngine(int dimx, int dimy, inputData<T> input, bool isGPU,int deviceId) : dimx(dimx), dimy(dimy), input(input), isGPU(isGPU), deviceId(deviceId)
{
    N=dimx*dimy;
    fAngle= atan2(dimy,dimx)*180/M_PI;

    //There will be an object of this class per thread, so,
    //this object will reuse intermediate storage for several angles

    //fAngle is the frontier:
    //Any angle in range [-bAngle,+bAngle] will use Normal (suffix N) storage, while...
    //any angle in range ]+bAngle,-bAngle[ will use Transposed (suffix T) storage

    //Largest dimension
    dim_l=MAX(dimx,dimy);

    weight=new float[dim_l]();
    target=new int[dim_l]();
    first=new short unsigned int[2*dim_l+1]();
    last=new short unsigned int[2*dim_l+1]();
    skewInput=new T[N*2+dim_l];
    skewOutput=new T[N*2+dim_l];
    if(isGPU)skewCudaMalloc();
}


/**
 * @tparam T
 */
template<typename T>
skewEngine<T>::~skewEngine() {
    if(isGPU)skewCudaFree();
    delete[] skewInput;
    delete[] skewOutput;
    delete[] target;
    delete[] weight;
    delete[] first;
    delete[] last;
    if(output0!= nullptr)delete[] output0;
    if(output1!= nullptr)delete[] output1;
    if(output2!= nullptr)delete[] output2;
    if(output3!= nullptr)delete[] output3;
}

template<typename T>
void skewEngine<T>::computeParams(int angle)
{
    a=angle;
    isT= angle > (int)fAngle && a < 180-(int)fAngle;
    isM= angle > 90;
    if(!isT&&!isM){sectorType=0;newAngle=a;}
    if( isT&&!isM){sectorType=1;newAngle=90-a;}
    if( isT&& isM){sectorType=2;newAngle=a-90;}
    if(!isT&& isM){sectorType=3;newAngle=180-a;}

    // This stands for deformation in surface measures
    //scale=1/pow(cos(torads*newAngle),2);
    scale=1/cos(torads*newAngle);

    // This is for drawing, only
    dim_skewx=isT?dimy:dimx;
    dim_skewy=isT?2*dimx:2*dimy;

    skewness=tan(newAngle*torads);
    iskewness=1/skewness;

    dim_i=isT?dimy:dimx; //inner dimension
    dim_o=isT?dimx:dimy; //outer dimension

    offset=skewness*(dim_i-1);
    skewHeight=newAngle?dim_o+offset+1:dim_o;//Excluded
    skewWidth=dim_i;

    //if(!isGPU)
    for (int j = 0; j < dim_i; j++) {
        double drift = skewness * j;
        target[j] = drift;
        weight[j] = drift - target[j]; //weight is lower when drift is close to target
    }

}

template<typename T>
void skewEngine<T>::resetSkew() {
    if(isGPU){
#if defined(__CUDACC__)
        cudaMemsetAsync(d_skewOutput, 0, dim_i* skewHeight* sizeof(*d_skewOutput), stream);
        cudaMemsetAsync(d_skewInput, 0, dim_i* skewHeight* sizeof(*d_skewInput), stream);
        //cudaStreamSynchronize(stream);
#endif
    }
    else {
        memset(skewInput, 0, skewHeight * dim_i * sizeof(*skewInput));
        memset(skewOutput, 0, skewHeight * dim_i * sizeof(*skewOutput));
    }
}

template<typename T>
void skewEngine<T>::skewShape()
{

    //Set all to zero for clean draw
    //for(int i=0;i<2*N+dim_i;i++)skewInput[i]=skewOutput[i]=0;
    //for(int i=0;i<=skewHeight*dim_i;i++)skewInput[i]=skewOutput[i]=0;

    for(int i=0;i<dim_o;i++)first[i]=0;
    for(int i=offset;i<skewHeight;i++)last[i]=dim_l;

    //temp values:
    for(int i=0;i<offset;i++)last[i]=0;
    for(int i=dim_o;i<skewHeight;i++)first[i]=dim_l;

    for (int j = 0; j < dim_i; j++) {
        int rows = target[j];
        if (j     > last[rows])last[rows] = j + 1;//+1 because excluded
        if (j     > last[rows+1])last[rows+1] = j + 1;
        int rowe = target[j] + dim_o-1;
        if (j     < first[rowe])first[rowe] = j;
        if (j     < first[rowe+1])first[rowe+1] = j;
    }

}

/**
 * Remap {image/dem/x-y data/etc.}  to its skewed position, calculates weights, and limits
 * Remenber: any pixel will be mapped to a couple of pixels, using the weights w and 1-w;
 * @tparam T
 * @param angle
 */
template<typename T>
void skewEngine<T>::skew(int angle){

    computeParams(angle);
    skewShape();
    resetSkew();

    if(!isGPU)
    {
        T *source = (T *) isT ? (isM ? input.input2 : input.input1) : (isM ? input.input3 : input.input0);
        for (int i = 0; i < dim_o; i++) {
            for (int j = 0; j < dim_i; j++) {
                skewInput[( i + target[j]  ) * dim_i + j]  += (1.0f - weight[j]) * source[dim_i * i+j];
                skewInput[( i + target[j]+1) * dim_i + j]  += weight[j] *          source[dim_i * i+j];
                }
        }
    }


#if defined(__CUDACC__)
    if(isGPU)
    {
        cudaMemcpyAsync(d_first, first, skewHeight* sizeof(*d_first), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_last, last, skewHeight* sizeof(*d_last), cudaMemcpyHostToDevice, stream);
        if(useV0)cudaMemcpyAsync(d_weight, weight, dim_i* sizeof(*d_weight), cudaMemcpyHostToDevice, stream);
        if(useV0)cudaMemcpyAsync(d_target, target, dim_i* sizeof(*d_target), cudaMemcpyHostToDevice, stream);
        //based on ExecuteAccMultiGPUmaster
        dim3 threadsPerBlock(8, 8,1);
        int gx = (dim_o % threadsPerBlock.x == 0) ? dim_o / threadsPerBlock.x : dim_o / threadsPerBlock.x + 1;
        int gy = (dim_i % threadsPerBlock.y == 0) ? dim_i / threadsPerBlock.y : dim_i / threadsPerBlock.y + 1;
        dim3 blocksPerGrid(gx, gy,1);
        if(useV0) cudaSkewV0<<< blocksPerGrid, threadsPerBlock, 0, stream >>>(d_skewInput,d_input+sectorType*N , skewness, dim_o, dim_i,d_weight,d_target);
        else cudaSkew<<< blocksPerGrid, threadsPerBlock, 0, stream >>>(d_skewInput,d_input+sectorType*N , skewness, dim_o, dim_i);
    }


#endif

}

/**
 * An intermediate (external) processing function will perform a line by line operation over skewed data
 * to generate skewed result:
 *
 * for(int i=lowerl; i< upperl; i++) //embarrassing parallel loop -> gpu candidate
 *    intensive CPU kernel for aligned data:
 *    for(int j=leftl(i); j< rightl(i); j++) ... skewOutput[i] = anytask(skewOutput[j])
 */


template<typename T>
void skewEngine<T>::kernel() {
    if(!isGPU) kernelcpu(this);
    else {
#if defined(__CUDACC__)

        dim3 threadsPerBlock(8, 8);
        int gx = (skewHeight % threadsPerBlock.x == 0) ? skewHeight / threadsPerBlock.x :
                 skewHeight / threadsPerBlock.x + 1;
        int gy = (dim_i % threadsPerBlock.y == 0) ? dim_i / threadsPerBlock.y : dim_i / threadsPerBlock.y + 1;
        dim3 blocksPerGrid(gx, gy);
        //cudaStreamSynchronize(stream);


            kernelgpu <<< blocksPerGrid, threadsPerBlock, 0, stream >>>(d_skewOutput,d_skewInput,dim_i,skewHeight,d_last,d_first,0.15f,newAngle);
#endif
    }
}


/**
 * After processing aligned data in the external algorithm (for example, total viewshed kernel,
 * this function maps-back skewed results (outputSkew)to its original position (output).
 * Now, the final pixel in output is a weighted average of skewed output data
 * @tparam T
 */
template<typename T>
void skewEngine<T>::deskew(int isMax){
    // https://stackoverflow.com/questions/7546620/operator-new-initializes-memory-to-zero
    // Un thread crea la estructura sólo si la va a necesitar
    if(sectorType==0&&output0== nullptr){
        output0=(T *)malloc(N*sizeof(T));
        memset((void *)output0,0,N* sizeof(T));
    }
    if(sectorType==1&&output1== nullptr){output1=(T *)malloc(N*sizeof(T));memset((void *)output1,0,N* sizeof(T));}
    if(sectorType==2&&output2== nullptr){output2=(T *)malloc(N*sizeof(T));memset((void *)output2,0,N* sizeof(T));}
    if(sectorType==3&&output3== nullptr){output3=(T *)malloc(N*sizeof(T));memset((void *)output3,0,N* sizeof(T));}
    T *output;
    if(sectorType==0)output=output0;
    if(sectorType==1)output=output1;
    if(sectorType==2)output=output2;
    if(sectorType==3)output=output3;

    if(!isGPU)
        switch(isMax){
        case 1:
            for(int i=0;i<dim_o;i++)
                for(int j=0;j<dim_i;j++) {
                    output[i * dim_i + j] = max(output[i * dim_i + j], max(skewOutput[(i + target[j]) * dim_i + j],
                                                                           skewOutput[(i + target[j] + 1) * dim_i +
                                                                                      j]));
                    //This line encodes angular information in less significative digits
                    //Output is supossed to va a number between 0 and 255000
                    output[i * dim_i + j]= a+ 1000*(int)(output[i * dim_i + j]/1000);

                }
            break;
        default:
            for(int i=0;i<dim_o;i++)
                for(int j=0;j<dim_i;j++)
                    output[i*dim_i+j]+=
                            (1.0f-weight[j])*skewOutput[(i+target[j]  )*dim_i+j]+
                            weight[j] *skewOutput[(i+target[j]+1)*dim_i+j];
            break;
    }


#if defined(__CUDACC__)
    if(isGPU) {
        dim3 threadsPerBlock(8, 8);
        int gx = (dim_o % threadsPerBlock.x == 0) ? dim_o / threadsPerBlock.x : dim_o / threadsPerBlock.x + 1;
        int gy = (dim_i % threadsPerBlock.y == 0) ? dim_i / threadsPerBlock.y : dim_i / threadsPerBlock.y + 1;
        dim3 blocksPerGrid(gx, gy);
        if(useV0) cudaDeskewV0<<< blocksPerGrid, threadsPerBlock >>>(d_output+sectorType*N, d_skewOutput, skewness, dim_o, dim_i, d_weight,d_target);
        else cudaDeskew<<< blocksPerGrid, threadsPerBlock,0,stream >>>(d_output+sectorType*N, d_skewOutput, skewness, dim_o, dim_i);
        cudaStreamSynchronize(stream);

    }
#endif
}

/**
 * Deprecated
 * Critical reduction of outputs from threads.
 * Any thread can have data of any of four kind of sectors
 * @tparam T
 * @param output
 */
template<typename T>
void skewEngine<T>::reduce(T **output) {
    for (int i = 0; i < dimy; i++){
        int ci=dimy-1-i;
        for (int j = 0; j < dimx; j++) {
            int cj=dimx-1-j;
            if (output0!= nullptr)output[i][j] += output0[i * dimx + j];
            if (output1!= nullptr)output[i][j] += output1[cj * dimy +ci];
            if (output2!= nullptr)output[i][j] += output2[j * dimy +ci];
            if (output3!= nullptr)output[i][j] += output3[i * dimx +cj];
        }
    }
}

#define REDUCE_ADD 0
#define REDUCE_MAX 1
#define REDUCE_MIN 2
#define REDUCE_IDXMAX 3
#define REDUCE_IDXMIN 4
template<typename T>
void skewEngine<T>::reduce(T *output, int isMax) {

#if defined(__CUDACC__)

    if(isGPU)   {
        if(output0!= nullptr)cudaMemcpyAsync(output0, d_output+0*N, N * sizeof(T), cudaMemcpyDeviceToHost, stream);
        if(output1!= nullptr)cudaMemcpyAsync(output1, d_output+1*N, N * sizeof(T), cudaMemcpyDeviceToHost, stream);
        if(output2!= nullptr)cudaMemcpyAsync(output2, d_output+2*N, N * sizeof(T), cudaMemcpyDeviceToHost, stream);
        if(output3!= nullptr)cudaMemcpyAsync(output3, d_output+3*N, N * sizeof(T), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
#endif
switch(isMax) {
    case 1:
        for (int i = 0; i < dimy; i++) {
            int ci = dimy - 1 - i;
            for (int j = 0; j < dimx; j++) {
                int cj = dimx - 1 - j;

                if (output0 != nullptr)output[i * dimx + j] =MAX(output[i * dimx + j], output0[i * dimx + j]);
                if (output1 != nullptr)output[i * dimx + j] =MAX(output[i * dimx + j], output1[cj * dimy + ci]);
                if (output2 != nullptr)output[i * dimx + j] =MAX(output[i * dimx + j], output2[j * dimy + ci]);
                if (output3 != nullptr)output[i * dimx + j] =MAX(output[i * dimx + j], output3[i * dimx + cj]);
            }
        }
        break;
    default:
        for (int i = 0; i < dimy; i++) {
            int ci = dimy - 1 - i;
            for (int j = 0; j < dimx; j++) {
                int cj = dimx - 1 - j;

                if (output0 != nullptr)output[i * dimx + j] += output0[i * dimx + j];
                if (output1 != nullptr)output[i * dimx + j] += output1[cj * dimy + ci];
                if (output2 != nullptr)output[i * dimx + j] += output2[j * dimy + ci];
                if (output3 != nullptr)output[i * dimx + j] += output3[i * dimx + cj];
            }
        }
        break;
}

}

/**
 * Static method to create the 4 versions of input data: NN, TN, TM, NM (allocating memory version)
 * @tparam T
 * @param inputD
 * @param dimx
 * @param dimy
 * @return
 */
template<typename T>
inputData<T> skewEngine<T>::prepare(T *inputD, int dimx, int dimy) {
    T *input0=inputD;
    T *input1=new T[dimx*dimy];
    T *input2=new T[dimx*dimy];
    T *input3=new T[dimx*dimy];

    // A transpose copy of input, which is computed once to improve locality
    for(int i=0;i<dimy;i++) {
        int ci=dimy-1-i;
        for (int j = 0; j < dimx; j++) {
            int cj=dimx - 1 - j;
            T val =
            input0[dimx * i +  j];
            input1[dimy *cj + ci] = val; //Blue arrow
            input2[dimy * j + ci] = val; //Black arrow
            input3[dimx * i + cj] = val;//Yellow arrow
        }
    }
    inputData<T> result;
    result.input0=input0;
    result.input1=input1;
    result.input2=input2;
    result.input3=input3;

    return result;

}

/**
 * Static function to create transposed/mirror inputs (pinned memory preallocated)
 * @tparam T
 * @param input
 * @param dimx
 * @param dimy
 * @return struct with four pointers
 */

template<typename T>
inputData<T> skewEngine<T>::prepare( inputData<T> *input , int dimx, int dimy) {

    // A transpose copy of input, which is computed once to improve locality
    T* input0 = input->input0;
    T* input1 = input->input1;
    T* input2 = input->input2;
    T* input3 = input->input3;
    for(int i=0;i<dimy;i++) {
        int ci=dimy-1-i;
        for (int j = 0; j < dimx; j++) {
            int cj=dimx - 1 - j;
            T val =
            input0[dimx * i +  j];
            input1[dimy *cj + ci] = val; //Blue arrow
            input2[dimy * j + ci] = val; //Black arrow
            input3[dimx * i + cj] = val;//Yellow arrow
        }
    }
    inputData<T> result;
    result.input0=input0;
    result.input1=input1;
    result.input2=input2;
    result.input3=input3;
    return result;
}




template<typename T>
void skewEngine<T>::skewCudaMalloc() {

#if defined(__CUDACC__)
    cudaError err = cudaSetDevice(deviceId);
    cudaStreamCreate(&stream);
    if(useV0)cudaMalloc(&d_target, dim_l * sizeof(int)) ;
    if(useV0)cudaMalloc(&d_weight, dim_l  * sizeof(double)) ;
    cudaMalloc(&d_first, (2*dim_l +1) * sizeof(unsigned short int)) ;
    cudaMalloc(&d_last, (2*dim_l +1) * sizeof(unsigned short int)) ;
    cudaMalloc(&d_input, 4 * dimy * dimx * sizeof(T)) ;
    cudaMalloc(&d_skewInput, (2 * dimy * dimx + dim_l) * sizeof(T));
    cudaMalloc(&d_skewOutput, (2 * dimy * dimx + dim_l) * sizeof(T));
    cudaMalloc(&d_output, 4 * dimy * dimx * sizeof(T));
    cudaMemsetAsync(d_output, 0, 4*dimy * dimx * sizeof(T), stream);
    cudaMemcpyAsync(d_input+0*dimx*dimy, input.input0, dimy * dimx * sizeof(T), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_input+1*dimx*dimy, input.input1, dimy * dimx * sizeof(T), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_input+2*dimx*dimy, input.input2, dimy * dimx * sizeof(T), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_input+3*dimx*dimy, input.input3, dimy * dimx * sizeof(T), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

#endif
}

template<typename T>
void skewEngine<T>::skewCudaFree() {
#if defined(__CUDACC__)
    //cudaStreamSynchronize(stream);
    if(useV0)cudaFree(d_target) ;
    if(useV0)cudaFree(d_weight) ;
    cudaFree(d_first) ;
    cudaFree(d_last) ;

    cudaFree(d_input) ;
    cudaFree(d_skewInput) ;
    cudaFree(d_skewOutput) ;
    cudaFree(d_output) ;
    cudaStreamDestroy(stream);
#endif
}


#endif
