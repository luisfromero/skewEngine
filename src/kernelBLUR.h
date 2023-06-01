/**
 * @file kernelBLUR.h
 * @author Felipe Romero
 * @brief Funciones específicas del kernel de cálculo del CEPSTRUM
 */

#ifndef KCEP_H
#define KCEP_H
#include "color.h"


#ifndef KERNELCOMMONS
#define KERNELCOMMONS

const int winSize=128; // Radio de la ventana

int isSamplePoint(int x , int y)
{
    if(saveSampleData)
        for(int i=0;i<samplePoints.size();i++)
            if(samplePoints[i].x==x&&samplePoints[i].y==y)return i;
    return -1;
}

void allocSampleData(int sector)
{
    for(int i=0;i<samplePoints.size();i++)
    {
        std::vector<float> v1(winSize);
        sampleData1[sector].push_back(v1);
        std::vector<float> v2(winSize/2);
        sampleData2[sector].push_back(v2);
        std::vector<float> v3(winSize/4);
        sampleData3[sector].push_back(v3);
    }
}

#endif





/**
 * Configuración específica para la transformada Radon
 * @param filename Nomvre del archivo de imagen, relativo a I_DIR, con extensión png incluida
 */
void configureCEPS(char *filename) {
    char fn[100];
    strcpy(fn,I_DIR);
    strcat(fn,filename);
    helper::read_png(fn,pixels,imgWidth,imgHeight);
    dimx=imgWidth;
    dimy=imgHeight;
    dim=N=dimx*dimy;
    inputD=new float[dim];
    for(int i=0;i<dim;i++)inputD[i]=pixels[i];
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
            subarray[j] = {v,0};
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

    if(saveSampleData){
        for(idxTrack=0;idxTrack<samplePoints.size();idxTrack++)
        {
            if(samplePoints[idxTrack].x==x&&samplePoints[idxTrack].y==y){
                found=true;
                break;
            }
        }
    }
    //if (last-first < windowSize)return 0;
    std::complex<double> input1[winSize];
    select_subarray(linePtr, input1, first, last, j, winSize,scale);
    CArray data1(input1, winSize);
    if(found)
    {
        for (int k = 0; k < winSize ; k++) sampleData1[angle][idxTrack][k]=((float)input1[k].real());
    }
    helper::fft(data1,true);
    float integral=0;
    int L=winSize/2;
    std::complex<double> input2[winSize];
    for (int k = 0; k < winSize/2 ; k++){
        double PS,logPS;
        PS=((pow(data1[k].real(), 2) + pow(data1[k].imag(), 2)));
        //PS=norm(data1[k]);
        logPS=log(PS);
        //integral+=logPS;
        input2[k] =std::complex<double>( logPS,0);
    }
    if(found)
    {
        for (int k = 0; k < winSize/2 ; k++) sampleData2[angle][idxTrack][k]=((float)input2[k].real());
    }
    CArray data2( input2, winSize/2);
    helper::fft(data2,true);
    double samples[winSize/4];
    for (int k = 0; k < winSize/4 ; k++)
    {
        samples[k] =-std::min(0.0,data2[k].real());
    }
    double result_val;
    getBlurParameter2(samples, winSize/4, result, result_val);
    integral=0;
    for (int k = 0; k < winSize/4 ; k++)integral+=samples[k];
    if(found)
    {
        for (int k = 0; k < winSize/4 ; k++) sampleData3[angle][idxTrack][k]=((float)(data2[k].real()));
    }
    //Draw cepstrum curves
    /*
    if(false) {
        if (i == 567 && j == 1066)
            for (int k = 0; k < winSize / 4; k++) {
                int y = std::max(0, (int) (30 - samples[k] / 1000));
                pointImgs[sector][y * 132 + 01 + k] = 0;
            }
        if (i == 138 && j == 357)
            for (int k = 0; k < winSize / 4; k++) {
                int y = std::max(0, (int) (30 - samples[k] / 1000));
                pointImgs[sector][y * 132 + 67 + k] = 0;
            }
    }
     */
    //if(result_val<2||result==0 || result == winSize/4)return 0;
    //float x=samples[result]-(samples[result-1]+samples[result+1])/2;
    //return integral/(scale*10);
    return result_val*10000;
    //if(result_val<2)result=0;
    //return result*3000;
    //result=  first_local_max(samples, winSize/4, result_val);
}


//Currently, thread unsafe
/*
float cepstrum_i(int i,int j, float *lineFirst,int length, fftw_plan p1, fftw_plan p2,fftw_complex *out,double *samples )
{
    //fftw_make_planner_thread_safe();
    int result;
    try {
        //fftw_complex *out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (winSize));
        //double samples[winSize];
        //fftw_plan p1,p2;

        if (length < winSize)return 0;
        select_subarray(lineFirst, samples, length, j, winSize);
        //p1 = fftw_plan_dft_r2c_1d(winSize, samples, out, FFTW_ESTIMATE);
        //if(p1==NULL)return 0;        else
            fftw_execute(p1);
        for (int k = 0; k < winSize / 2 + 1; k++)samples[k] = log((pow(out[k][0], 2) + pow(out[k][1], 2)));
        //p2 = fftw_plan_dft_r2c_1d(winSize / 2 + 1, samples, out, FFTW_ESTIMATE);
        //if(p2==NULL)return 0;         else
            fftw_execute(p2);
        for (int k = 0; k < (winSize / 2 + 1) / 2 + 1; k++)samples[k] = (pow(out[k][0], 2) + pow(out[k][1], 2));
        //fftw_free(out);
        //fftw_destroy_plan(p1);
        //fftw_destroy_plan(p2);
        double v;
        result=  first_local_max(samples, (winSize/2+1)/2+1,v);
    }catch(std::exception e)
    {
        return 0;
    }
    return (float)result;
}
*/



void cepstrum(skewEngine<float> *skewEngine)
{
    //fftw_complex *out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (winSize));
    //double samples[winSize];
    //fftw_make_planner_thread_safe();
    //fftw_complex *out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (winSize));
    //double samples[winSize];
    //fftw_plan p1,p2;
    //p1 = fftw_plan_dft_r2c_1d(winSize, samples, out, FFTW_ESTIMATE);
    //p2 = fftw_plan_dft_r2c_1d(winSize / 2 + 1, samples, out, FFTW_ESTIMATE);

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



void showResultsCEPS()
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
        for (int j = 0; j < winSize/2; j++) {
            punto1i << sampleData1[i][0][winSize/2+j] << " ";
            punto2i << sampleData1[i][1][winSize/2+j] << " ";
            punto3i << sampleData1[i][2][winSize/2+j] << " ";
        }
        punto1i << std::endl;
        punto2i << std::endl;
        punto3i << std::endl;
    }
    for (int i = 0; i < 180; i++) {
        for (int j = 0; j < winSize/2; j++) {
            punto1i << sampleData1[i][0][winSize/2-1-j] << " ";
            punto2i << sampleData1[i][1][winSize/2-1-j] << " ";
            punto3i << sampleData1[i][2][winSize/2-1-j] << " ";
        }
        punto1i << std::endl;
        punto2i << std::endl;
        punto3i << std::endl;
    }

    // Escribir segundo bloque en punto1_k1.txt
    for (int i = 0; i < 180; i++) {
        for (int j = 0; j < winSize/2; j++) {
            punto1l << sampleData2[i][0][j] << " ";
            punto2l << sampleData2[i][1][j] << " ";
            punto3l << sampleData2[i][2][j] << " ";
        }
        punto1l << std::endl;
        punto2l << std::endl;
        punto3l << std::endl;
    }
    for (int i = 0; i < 180; i++) {
        for (int j = 0; j < winSize/2; j++) {
            punto1l << sampleData2[i][0][j] << " ";
            punto2l << sampleData2[i][1][j] << " ";
            punto3l << sampleData2[i][2][j] << " ";
        }
        punto1l << std::endl;
        punto2l << std::endl;
        punto3l << std::endl;
    }

    // Escribir tercer bloque en punto2_k2.txt
    for (int i = 0; i < 180; i++) {
        for (int j = 0; j < winSize/4; j++) {
            punto1c << sampleData3[i][0][j] << " ";
            punto2c << sampleData3[i][1][j] << " ";
            punto3c << sampleData3[i][2][j] << " ";
        }
        punto1c << std::endl;
        punto2c << std::endl;
        punto3c << std::endl;
    }
    for (int i = 0; i < 180; i++) {
        for (int j = 0; j < winSize/4; j++) {
            punto1c << sampleData3[i][0][j] << " ";
            punto2c << sampleData3[i][1][j] << " ";
            punto3c << sampleData3[i][2][j] << " ";
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