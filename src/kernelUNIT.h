/**
 * @file kernelUNIT.h
 * @author Felipe Romero
 * @brief Specific functions of the Identity kernel
 *
 * Modelo para otros kernel
 */

#ifndef KUNIT_H
#define KUNIT_H


void configureUNIT(char *filename) {
}

void identity(skewEngine<float> *skewEngine)
{
    for(int i=0;i<skewEngine->skewHeight;i++){
        int k=skewEngine->skewWidth*i;
        for(int j=skewEngine->first[i];j<skewEngine->last[i];j++)
            skewEngine->skewOutput[k+j]=skewEngine->skewInput[k+j];
    }
}
__global__
void identityCuda(float *d_skewOutput, float *d_skewInput, int dim_i, int skewHeight,  unsigned short * d_last,unsigned short * d_first, float val, int ang) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    //int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row>=skewHeight)return;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    //int col = blockIdx.y * blockDim.y + threadIdx.y;
    int start, end;
    start=d_first[row];
    end=d_last[row];
    if(col>=end&&col<start)return;
    int idx= row * dim_i + col;
    d_skewOutput[idx]=d_skewInput[idx];
    return;
}


const std::string identityOCL=
        "__kernel void mainKernel(global float* T, global float *S, int w, int h, global unsigned short *last, global unsigned short *first, float val, int ang)"
        "   {\n"
        "//    int row = get_group_id(0) * get_local_size(0) + get_local_id(0);\n"
        "    int row = get_group_id(1) * get_local_size(1) + get_local_id(1);\n"
        "//    if(row>h)return;\n"
        "//    int col = get_group_id(1) * get_local_size(1) + get_local_id(1);\n"
        "    int col = get_group_id(0) * get_local_size(0) + get_local_id(0);\n"
        "//    if(col<first[row])return;\n"
        "//    if(col>=last[row])return;\n"
        "    int i=row*w+col;\n"
        "       T[i]=S[i];\n"
        "   }\n";


void showResultsUNIT()
{
    //Aquí deberíamos calcular el porcentaje de diferencias debido a la interpolación y extrapolación
}

#endif