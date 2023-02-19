/**
 * @file kernelRADN.h
 * @author Felipe Romero
 * @brief Funciones específicas del kernel para la transformada RADON
 */

#ifndef KRND_H
#define KRND_H
#include "../color.h"
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


void configureRADN(char *filename) {
    char fn[100];
    strcpy(fn,I_DIR);
    strcat(fn,filename);
    read_png(fn,pixels,imgWidth,imgHeight);
    dimx=imgWidth;
    dimy=imgHeight;
    dim=N=dimx*dimy;
    inputD=new float[dim];
    for(int i=0;i<dim;i++)inputD[i]=pixels[i];
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



void showResultsRADN()
{
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


    std::vector<unsigned char> grey;
    std::vector<unsigned char> test;
    for(int i=0;i<dim;i++) {
        grey.push_back(pixels[i]/3);
        rgbColor c;
        if(skewAlgorithm==3) c={(unsigned char)((int)outD[i]/540),(unsigned char)((int)outD[i]/540),(unsigned char)((int)outD[i]/540)};
        if(skewAlgorithm==1) c=HSVtoRGB({static_cast<float>(((int)outD[i])%360), 100.0f, std::min(100.0f,outD[i]/3000.0f)});  //50 for cepst index of peak
        if(skewAlgorithm==4)
            c=HSVtoRGB({static_cast<float>(((int)outD[i])%360), 100.0f, std::min(100.0f,outD[i]/(180*3000.0f))});  //50 for cepst index of peak
        test.push_back(c.R);
        test.push_back(c.G);
        test.push_back(c.B);
    }
    lodepng::encode("salida1.png", grey, dimx, dimy, LCT_GREY);
    lodepng::encode("salida2.png", test, dimx, dimy, LCT_RGB);


}


#endif