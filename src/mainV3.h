//#include "fft/fftw3.h"
#include <complex>
#include <iostream>
#include <valarray>
#include <algorithm>


//*********************************************************************************************
//*********************************************************************************************
//                                 Kernels for CPU mode (kernelV3 -> sDEM)
//*********************************************************************************************
//*********************************************************************************************

// These kernels have an outer loop from 0 to skewHeight, and inner loop from first to last,
// usually from 0 to din_i (inner)


const int windowSize=256;
typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;

int nPoints=3;
const bool saveTrackData=true;
//point_t punto1={3,4};
//point_t punto2={103,104};
//point_t punto3={223,224};

point_t punto1={1082,562};
point_t punto2={348,266};
point_t punto3={600,536};

std::vector<point_t>  trackPoints;
std::vector<std::vector<float>>  trackData1[180];
std::vector<std::vector<float>>  trackData2[180];
std::vector<std::vector<float>>  trackData3[180];
void allocTrackData(int sector)
{
    for(int i=0;i<trackPoints.size();i++)
    {
        std::vector<float> v1(windowSize);
        trackData1[sector].push_back(v1);
        std::vector<float> v2(windowSize/2);
        trackData2[sector].push_back(v2);
        std::vector<float> v3(windowSize/4);
        trackData3[sector].push_back(v3);
    }
}




// Cooley-Tukey FFT (in-place, breadth-first, decimation-in-frequency)
// Better optimized but less intuitive
// !!! Warning : in some cases this code make result different from not optimased version above (need to fix bug)
// The bug is now fixed @2017/05/30
void fft(CArray &x, bool normalize=false)
{
    // DFT
    unsigned int N = x.size(), k = N, n;
    double thetaT = 3.14159265358979323846264338328L / N;
    Complex phiT = Complex(cos(thetaT), -sin(thetaT)), T;
    while (k > 1)
    {
        n = k;
        k >>= 1;
        phiT = phiT * phiT;
        T = 1.0L;
        for (unsigned int l = 0; l < k; l++)
        {
            for (unsigned int a = l; a < N; a += n)
            {
                unsigned int b = a + k;
                Complex t = x[a] - x[b];
                x[a] += x[b];
                x[b] = t * T;
            }
            T *= phiT;
        }
    }
    // Decimate
    unsigned int m = (unsigned int)log2(N);
    for (unsigned int a = 0; a < N; a++)
    {
        unsigned int b = a;
        // Reverse bits
        b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
        b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
        b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
        b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
        b = ((b >> 16) | (b << 16)) >> (32 - m);
        if (b > a)
        {
            Complex t = x[a];
            x[a] = x[b];
            x[b] = t;
        }
    }
    //// Normalize (This section make it not working correctly)
    if(!normalize)return;
    Complex f = 1.0 / sqrt(N);
    for (unsigned int i = 0; i < N; i++)
    	x[i] *= f;
}


void select_subarray(float array[], double subarray[], int first, int last, int i, int winSize) {
    int half = winSize / 2;
    for (int j = 0; j < winSize; j++) {
        int index = i - half + j;
        if (index < first || index >= last) {
            subarray[j] = 0;
        } else {
            subarray[j] = array[index];
        }
    }
}

/**
 * Select Subset of winSize data from row , centered at i
 * @param array Row of data
 * @param subarray Target storage
 * @param size Right limit
 * @param i
 * @param winSize Subset size
 */
void select_subarray(float array[], Complex *subarray, int first, int last, int i, int winSize, float sigma=1) {
    int half = winSize / 2;
    double half1 = (winSize-1) / 2.0;
    sigma= sigma*winSize / 20;
    for (int j = 0; j < winSize; j++) {
        int index = i - half + j;
        if (index < first || index >= last) {
            subarray[j] = {0,0};
        } else {
            float v=array[index];
            double gauss = exp(-(j - half) * (j - half) / (2 * sigma * sigma));
            double alpha = (j - half1 / half1);
            double hahn=0.5 * (1.0 - cos(2.0*M_PI*alpha));
            subarray[j] = {v*hahn,0};
        }
    }
}

//Absolute max
void getBlurParameter2(double *v, int n, int &imax, double &val)
{
    val=0;
    imax=0;
    int i;
    //Mientras sea descendiente, descartamos
    for(i=5;i<n;i++)
        if(v[i]>v[i-1])
            break;
    if(i==n)
        return;
    else
    {val=v[i-1];imax=i++;}

    for(;i<n;i++)
        if(v[i]>val){
            val=v[i];
            imax=i;

        }
    if(imax==0)
        val=0;
}

//First max
void getBlurParameter(double *v, int n, int &imin, double &val)
{
    val=0;
    imin=0;
    int i;
    for(i=4;i<n-1;i++)
        if(v[i]<0&&(v[i]<v[i+1])&&(v[i]<v[i-1]))
        {
            imin=i;
            val=-v[i];
        }
}
//First max
void getBlurParameter3(double *v, int n, int &imax, double &val)
{
    val=0;
    imax=0;
    int i;
    //Mientras sea descendiente, descartamos
    for(i=4;i<n;i++)
        if(v[i]>v[i-1])
            break;
    if(i==n)
        return;
    else
    {val=v[i-1];imax=i++;}

    for(;i<n-1;i++)
        if(v[i]>v[i+1]){
            val=v[i];
            imax=i;
            return;
        }
    if(imax==0)
        val=0;
}



//Simpler and thread safe, and no lib required
float lineCepstrum(int j, float *linePtr,int first, int last, int x, int y, int angle, float scale=1) {
    int result;
    int idxTrack;
    bool found=false;

    if(saveTrackData){
        for(idxTrack=0;idxTrack<trackPoints.size();idxTrack++)
        {
            if(trackPoints[idxTrack].x==x&&trackPoints[idxTrack].y==y){
                found=true;
                break;
            }
        }
    }
    //if (last-first < windowSize)return 0;
    std::complex<double> input1[windowSize];
    select_subarray(linePtr, input1, first, last, j, windowSize,scale);
    CArray data1(input1, windowSize);
    if(found)
    {
        for (int k = 0; k < windowSize ; k++) trackData1[angle][idxTrack][k]=((float)input1[k].real());
    }
    fft(data1,true);
    float integral=0;
    int L=windowSize/2;
    std::complex<double> input2[windowSize];
    for (int k = 0; k < windowSize/2 ; k++){
        double PS,logPS;
        PS=((pow(data1[k].real(), 2) + pow(data1[k].imag(), 2)));
        //PS=norm(data1[k]);
        logPS=log(PS);
        //integral+=logPS;
        input2[k] =std::complex<double>( logPS,0);
    }
    if(found)
    {
        for (int k = 0; k < windowSize/2 ; k++) trackData2[angle][idxTrack][k]=((float)input2[k].real());
    }
    CArray data2( input2, windowSize/2);
    fft(data2,true);
    double samples[windowSize/4];
    for (int k = 0; k < windowSize/4 ; k++)
    {
        samples[k] =-std::min(0.0,data2[k].real());
    }
    double result_val;
    getBlurParameter2(samples, windowSize/4, result, result_val);
    integral=0;
    for (int k = 0; k < windowSize/4 ; k++)integral+=samples[k];
    if(found)
    {
        for (int k = 0; k < windowSize/4 ; k++) trackData3[angle][idxTrack][k]=((float)(data2[k].real()));
    }
    //Draw cepstrum curves
    /*
    if(false) {
        if (i == 567 && j == 1066)
            for (int k = 0; k < windowSize / 4; k++) {
                int y = std::max(0, (int) (30 - samples[k] / 1000));
                pointImgs[sector][y * 132 + 01 + k] = 0;
            }
        if (i == 138 && j == 357)
            for (int k = 0; k < windowSize / 4; k++) {
                int y = std::max(0, (int) (30 - samples[k] / 1000));
                pointImgs[sector][y * 132 + 67 + k] = 0;
            }
    }
     */
    //if(result_val<2||result==0 || result == windowSize/4)return 0;
    //float x=samples[result]-(samples[result-1]+samples[result+1])/2;
    //return integral/(scale*10);
    return result_val*10000;
    //if(result_val<2)result=0;
    //return result*3000;
    //result=  first_local_max(samples, windowSize/4, result_val);
}


//Currently, thread unsafe
/*
float cepstrum_i(int i,int j, float *lineFirst,int length, fftw_plan p1, fftw_plan p2,fftw_complex *out,double *samples )
{
    //fftw_make_planner_thread_safe();
    int result;
    try {
        //fftw_complex *out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (windowSize));
        //double samples[windowSize];
        //fftw_plan p1,p2;

        if (length < windowSize)return 0;
        select_subarray(lineFirst, samples, length, j, windowSize);
        //p1 = fftw_plan_dft_r2c_1d(windowSize, samples, out, FFTW_ESTIMATE);
        //if(p1==NULL)return 0;        else
            fftw_execute(p1);
        for (int k = 0; k < windowSize / 2 + 1; k++)samples[k] = log((pow(out[k][0], 2) + pow(out[k][1], 2)));
        //p2 = fftw_plan_dft_r2c_1d(windowSize / 2 + 1, samples, out, FFTW_ESTIMATE);
        //if(p2==NULL)return 0;         else
            fftw_execute(p2);
        for (int k = 0; k < (windowSize / 2 + 1) / 2 + 1; k++)samples[k] = (pow(out[k][0], 2) + pow(out[k][1], 2));
        //fftw_free(out);
        //fftw_destroy_plan(p1);
        //fftw_destroy_plan(p2);
        double v;
        result=  first_local_max(samples, (windowSize/2+1)/2+1,v);
    }catch(std::exception e)
    {
        return 0;
    }
    return (float)result;
}
*/



void cepstrum(skewEngine<float> *skewEngine)
{
    //fftw_complex *out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (windowSize));
    //double samples[windowSize];
    //fftw_make_planner_thread_safe();
    //fftw_complex *out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (windowSize));
    //double samples[windowSize];
    //fftw_plan p1,p2;
    //p1 = fftw_plan_dft_r2c_1d(windowSize, samples, out, FFTW_ESTIMATE);
    //p2 = fftw_plan_dft_r2c_1d(windowSize / 2 + 1, samples, out, FFTW_ESTIMATE);

    int q=skewEngine->sectorType;
    bool tr,rx,ry;
    float scale =1/skewEngine->scale;
    for(int i=0;i<skewEngine->skewHeight;i++){
        int k=skewEngine->skewWidth*i;
        int length=skewEngine->last[i]-skewEngine->first[i];
        for(int j=skewEngine->first[i];j<skewEngine->last[i];j++) {
            // x & y as source data
            if(q==0){tr=false;rx=false;ry=false;};
            if(q==1){tr=true;rx=true;ry=true;};
            if(q==2){tr=true;rx=false;ry=true;};
            if(q==3){tr=false;rx=true;ry=false;};

            int y=!tr?i-skewEngine->target[j]:j;
            int x=!tr?j:i-skewEngine->target[j];
            x=rx?dimx-1-x:x;
            y=ry?dimy-1-y:y;
            skewEngine->skewOutput[k + j] =
                    lineCepstrum(j, &skewEngine->skewInput[k], skewEngine->first[i], skewEngine->last[i],
                                 x, y,skewEngine->a, scale);
            //skewEngine->skewOutput[k+j]= cepstrum_i(i,j,&skewEngine->skewInput[k],length,p1,p2,out,samples);
        }}
    //fftw_free(out);
    //fftw_cleanup();
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

void kernelV3(skewEngine<float> *skewer)
{
    for (int i = 0; i < skewer->skewHeight; i++) {
        float r =skewer->scale;
        int inicio= skewer->first[i];
        int final = skewer->last[i];

        float cv,  max_angle, open_delta_d, delta_d, cAngle;
        float h;
        bool visible, above, opening, closing;
        int rowptr= skewer->skewWidth * i;

        for (int j = inicio; j < final; j++) {

            cv = 0;
            h = skewer->skewInput[rowptr + j]+POVh;

            max_angle = -2000;
            visible = true;
            open_delta_d = 0;

            for (int k = j + 1; k >= inicio && k < final; k++) {
                delta_d = k - j;
                cAngle = (skewer->skewInput[rowptr + k] - h) / delta_d;
                above = (cAngle > max_angle);
                opening = above && (!visible);
                closing = (!above) && (visible);
                visible = above;
                max_angle = std::max(cAngle, max_angle);
                if (opening) open_delta_d = delta_d;
                if (closing) cv += (delta_d * delta_d - open_delta_d * open_delta_d);
            }

            max_angle = -2000;
            visible = true;
            open_delta_d = 0;

            for (int k = j - 1; k >= inicio && k < final; k--) {
                delta_d = j - k;
                //float idelta=_mm_cvtss_f32(_mm_rcp_ss( _mm_set_ss((int)delta_d) ));
                //cAngle = (skwDEM[i][k] - h) * idelta ;
                cAngle = (skewer->skewInput[rowptr + k] - h) /delta_d;
                above = (cAngle > max_angle);
                opening = above && (!visible);
                closing = (!above) && (visible);
                visible = above;
                max_angle = std::max(cAngle, max_angle);
                if (opening) open_delta_d = delta_d;
                if (closing) cv += (delta_d * delta_d - open_delta_d * open_delta_d);
            }
            skewer->skewOutput[rowptr + j] = cv * r;
        }
    }

}
__global__
void kernelV3cuda(float *skewOutput, float *skewInput, int dim_i, int skewHeight, unsigned short *d_last, unsigned short *d_first, float POVh, int angle) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= skewHeight)return;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int start, end;
    start=d_first[row];
    end=d_last[row];
    if(col>=end&&col<start)return;
    float torads=M_PI/180;
    float inrads=angle*torads;
    float r = 1.0 / pow(cos(inrads),2);


    float cv, h, max_angle, open_delta_d, delta_d, cAngle;
    bool visible, above, opening, closing;

//    if (row < endGPUbatch && col >= start && col < end)
    {
        cv = 0;                                                                                 // Ya tengo observador en j
        h = skewInput[row * dim_i + col] + POVh;
        max_angle = -2000;
        visible = true;
        open_delta_d = 0;

        for (int k = col + 1; k >= start && k < end; k++) {                                       // Calculo cv a dcha

            delta_d = k - col;
            cAngle = (skewInput[row * dim_i + k] - h) / delta_d;
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

        for (int k = col - 1; k >= start && k < end; k--) {                                      // Calculo cv a izqda

            delta_d = col - k;
            cAngle = (skewInput[row * dim_i + k] - h) / delta_d;
            above = (cAngle > max_angle);
            opening = above && (!visible);
            closing = (!above) && (visible);

            visible = above;
            max_angle = max(cAngle, max_angle);
            if (opening) open_delta_d = delta_d;
            if (closing) cv += (delta_d * delta_d - open_delta_d * open_delta_d);
        }

        skewOutput[row * dim_i + col] = cv * r;
    }

}



//*********************************************************************************************
//*********************************************************************************************

/**
 * Función principal
 * Generico para CPU, GPU o híbrido
 */


void executeV3(int skewAlgorithm=0)
{
//inData is the input data
//inputD is a struct with 4 pointers (4 arrays of input data: NN, TM, TN, TM)

//    inputData<float> inData=  skewEngine<float>::prepare(inputD,dimx,dimy);//Create rotated and mirror versions of input
    bool ident=skewAlgorithm==2||skewAlgorithm==3;
    //omp_set_num_threads(1);
    trackPoints.push_back(punto1);
    trackPoints.push_back(punto2);
    trackPoints.push_back(punto3);

    for (int i = 0; i < 180; i++)allocTrackData(i);


#pragma omp parallel default(none) shared(inData,dimx,dimy,runMode,outD,maxSector,ident,skewAlgorithm)
    {
        int id = omp_get_thread_num();

// Each thread (in CPU mode -> arbitrary)  (in GPU mode -> nthreads = num of GPUs) has its own engine:
        skewEngine<float> *skewer=new skewEngine<float>(dimx, dimy, static_cast<inputData<float>>(inData), runMode == 1,id);
#pragma omp barrier
#pragma omp for schedule(dynamic) nowait
        for (int i = 0; i < 180; i++) {


            skewer->skew(i);
            switch(skewAlgorithm) {
                case 0:
                    skewer->kernelcpu = kernelV3;
                    skewer->kernelgpu =kernelV3cuda;
                    break;
                case 1:
                    // Selecciono el máximo
                    skewer->kernelcpu = cepstrum;
                    //ToDo Versión blur para CUDA
                    break;
                case 4:
                    // Acumula
                    skewer->kernelcpu = cepstrum;
                    //ToDo Versión blur para CUDA
                    break;
                default:
                    skewer->kernelcpu = identity;
                    skewer->kernelgpu =identityCuda;
                    break;
            }


            skewer->kernel();

            skewer->deskew(skewAlgorithm==1?   1:0);
            printf("id= %03d se=%03d\n",id,i);//fflush(stdout);
        }

#pragma omp critical
        {
            // When finishing, thread data are added to outD
            skewer->reduce(outD,skewAlgorithm==1?1:0);
        }
        delete skewer;



    } //end parallel region



}
void showResults(int skewAlgorithm) {
    pair_t mm = getMinMax(outD);
    float escala = 1.0 / 180;
    if (skewAlgorithm == 0)escala = surScale; //scales to hectometers
    if (skewAlgorithm == 1)escala = 1;//1.0/180;  //cepstrum
    if (skewAlgorithm == 2)escala = 10.0 / 180; //identity sdem
    if (skewAlgorithm == 3)escala = 1.0 / 540; //identity blur
    if (skewAlgorithm == 4)escala = 1.0 / 180;//1.0/180;  //cepstrum

    printf("Extreme values for output: %6.2f - %e  (scale = %f )\n ", (mm.min * escala), mm.max * escala, escala);
    fflush(stdout);


    std::ofstream punto1i("punto1i.txt");
    std::ofstream punto2i("punto2i.txt");
    std::ofstream punto3i("punto3i.txt");
    std::ofstream punto1l("punto1l.txt");
    std::ofstream punto2l("punto2l.txt");
    std::ofstream punto3l("punto3l.txt");
    std::ofstream punto1c("punto1c.txt");
    std::ofstream punto2c("punto2c.txt");
    std::ofstream punto3c("punto3c.txt");

    for (int i = 0; i < 180; i++) {
        for (int j = 0; j < windowSize/2; j++) {
            punto1i << trackData1[i][0][windowSize/2+j] << " ";
            punto2i << trackData1[i][1][windowSize/2+j] << " ";
            punto3i << trackData1[i][2][windowSize/2+j] << " ";
        }
        punto1i << std::endl;
        punto2i << std::endl;
        punto3i << std::endl;
    }
    for (int i = 0; i < 180; i++) {
        for (int j = 0; j < windowSize/2; j++) {
            punto1i << trackData1[i][0][windowSize/2-1-j] << " ";
            punto2i << trackData1[i][1][windowSize/2-1-j] << " ";
            punto3i << trackData1[i][2][windowSize/2-1-j] << " ";
        }
        punto1i << std::endl;
        punto2i << std::endl;
        punto3i << std::endl;
    }

    // Escribir segundo bloque en punto1_k1.txt
    for (int i = 0; i < 180; i++) {
        for (int j = 0; j < windowSize/2; j++) {
            punto1l << trackData2[i][0][j] << " ";
            punto2l << trackData2[i][1][j] << " ";
            punto3l << trackData2[i][2][j] << " ";
        }
        punto1l << std::endl;
        punto2l << std::endl;
        punto3l << std::endl;
    }
    for (int i = 0; i < 180; i++) {
        for (int j = 0; j < windowSize/2; j++) {
            punto1l << trackData2[i][0][j] << " ";
            punto2l << trackData2[i][1][j] << " ";
            punto3l << trackData2[i][2][j] << " ";
        }
        punto1l << std::endl;
        punto2l << std::endl;
        punto3l << std::endl;
    }

    // Escribir tercer bloque en punto2_k2.txt
    for (int i = 0; i < 180; i++) {
        for (int j = 0; j < windowSize/4; j++) {
            punto1c << trackData3[i][0][j] << " ";
            punto2c << trackData3[i][1][j] << " ";
            punto3c << trackData3[i][2][j] << " ";
        }
        punto1c << std::endl;
        punto2c << std::endl;
        punto3c << std::endl;
    }
    for (int i = 0; i < 180; i++) {
        for (int j = 0; j < windowSize/4; j++) {
            punto1c << trackData3[i][0][j] << " ";
            punto2c << trackData3[i][1][j] << " ";
            punto3c << trackData3[i][2][j] << " ";
        }
        punto1c << std::endl;
        punto2c << std::endl;
        punto3c << std::endl;
    }


        // Cerrar archivos
    punto1i.close();
    punto2i.close();
    punto3i.close();
    punto1l.close();
    punto2l.close();
    punto3l.close();
    punto1c.close();
    punto2c.close();
    punto3c.close();




}
