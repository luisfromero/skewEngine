/**
 * @file kernelSDEM.h
 * @author Felipe Romero
 * @brief Funciones específicas del kernel unitario
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
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row>=skewHeight)return;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int start, end;
    start=d_first[row];
    end=d_last[row];
    if(col>=end&&col<start)return;
    int idx= row * dim_i + col;
    d_skewOutput[idx]=d_skewInput[idx];
    return;
}


void showResultsUNIT()
{
}

#endif