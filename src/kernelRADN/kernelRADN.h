/**
 * @file kernelRADN.h
 * @author Felipe Romero
 * @brief Funciones específicas del kernel para la transformada RADON
 */

#ifndef KRND_H
#define KRND_H
#include "../color.h"

#ifndef M_PI_2
#define M_PI_2 (M_PI/2)
#endif

#ifndef KERNELCOMMONS
#define KERNELCOMMONS

const int winSize=64; // Diámetro de la ventana

/**
 * Obtiene las coordenadas originales de un mapa sesgado
 * @param i Row, in skewed map
 * @param j Column, in skewed map
 * @param q Quadrant
 * @return Coordinates x and y in the original 2D data
 */
point_t getXY(int i, int j, skewEngine<float> *sk)
{
    bool tr,rx,ry;
    int q=sk->sectorType;
    if(q==0){tr=false;rx=false;ry=false;};
    if(q==1){tr=true;rx=true;ry=true;};
    if(q==2){tr=true;rx=false;ry=true;};
    if(q==3){tr=false;rx=true;ry=false;};
    int y=!tr?i-sk->target[j]:j;
    int x=!tr?j:i-sk->target[j];
    x=rx?dimx-1-x:x;
    y=ry?dimy-1-y:y;
    return {x,y};
}


int isSamplePoint(int x , int y)
{
    if(saveSampleData)
        for(int i=0;i<samplePoints.size();i++)
            if(samplePoints[i].x==x&&samplePoints[i].y==y)return i;
    return -1;
}
int isSamplePoint(point_t pt)
{
    return isSamplePoint(pt.x,pt.y);
}

void allocSampleData(int sector)
{
    for(int i=0;i<samplePoints.size();i++)
    {
        std::vector<float> v1(winSize);
        sampleData1[sector].push_back(v1);
        std::vector<float> v2(winSize);
        sampleData2[sector].push_back(v2);
        std::vector<float> v3(winSize);
        sampleData3[sector].push_back(v3);
    }
}

#endif


/**
 * Configuración específica para la transformada Radon
 * @param filename Nombre del archivo de imagen, relativo a I_DIR, con extensión png incluida
 */
void configureRADN(char *filename) {
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
 * This method is called for every point in every skewed mapping.
 * @param i Row index in skewed mapping
 * @param j Column index in skewed mapping
 * @param sk A pointer to the skewEngine (thread private)
 * @param x Row index in the original data
 * @param y Column index in the original data
 * @returns Computed value for this point
 */
float lineRadon(int i,int j, skewEngine<float> * sk) {
    /// Averiguamos si es un punto de interés. Devuelve -1 en caso contrario
    int idxTrack=isSamplePoint(getXY(i,j,sk));

    int w=sk->skewWidth;
    int h=sk->skewHeight;
    int w2=winSize/2;
    double dws=winSize;
    double dw2=w2;
    double dw1 = (winSize-1);
    double sigma= sigma*winSize / 20;
    double cutoff=0.5;
    /// Calculate Radon transform (ramp filtered)
    std::complex<double> input1[winSize];
    for(int k=i-w2;k<i+w2;k++) {  //Par. Punto i en posición w2 de input1
        double accum=0;
        int fi=sk->first[k];
        int la=sk->last[k];
        for (int l = j-w2; l <= j+w2; l++) //Impar. Centrado en j
            accum += (k<0|| k>=h)?0:( (l<fi||l>=la)?0:sk->skewInput[k*w+l] );
        int kk=k - (i-w2); //de 0 a wsize
        double gauss = exp(-(j - w2) * (j - w2) / (2 * sigma * sigma));
        double hahn=0.5 * (1.0 - cos(2.0*M_PI*(kk / dw1)));// 1-cos -> 0 a 2 a 0 ->
        double ramp=1-abs((k-i)/dw2);
        input1[k - (i-w2)]= {accum*ramp,0};
    }

    /// FFT to Radon
    CArray data1(input1, winSize);
    helper::fft(data1,true);
    std::complex<double> input2[winSize];

    //Frecuencia 0 en k=0. En k=winSize, la frecuencia es la de muestreo
    //Se aplica un filtro pasa-alta
    for(int k=0;k<winSize;k++) {
        // input2[k] = k<5?0:data1[k].real(); //Filtro pasa alta alo bruto -> borrosillo
        // input2[k] = data1[k].real(); //No filtramos nada -> borroso
        // input2[k] = sin((k/dw1) *M_PI_2)* data1[k].real(); //Shepp-Logan de 0 a 1 a ritmo de Pi

        input2[k] = k> cutoff*dws?0:sin((k/(cutoff*dws-1)) *M_PI_2)* data1[k].real(); //Shepp-Logan cutoff at 80%
        //input2[k] = (k/dw1)* data1[k].real(); //Filtro rampa
    }
    CArray data2(input2, winSize);
    helper::fft(data2,true);
    //helper::fft(data1,true);

    if(idxTrack>=0)
    {
        for (int k = 0; k < winSize ; k++) sampleData1[sk->a][idxTrack][k]=((float)input1[k].real());
        for (int k = 0; k < winSize ; k++) sampleData2[sk->a][idxTrack][k]=((float)input2[k].real());
        for (int k = 0; k < winSize/4 ; k++) sampleData3[sk->a][idxTrack][k]=((float)data2[k].real());
    }

    std::complex<double> val1=data2[w2].real();
    //std::complex<double> val2=data1[winSize / 2].real()/winSize;
    double val0=sk->skewInput[i*w+j];



    return val1.real();//<0?0:val1.real() ;
}


void radon(skewEngine<float> *skewEngine)
{
    for(int i=0;i<skewEngine->skewHeight;i++){
        int k=skewEngine->skewWidth*i;
        for(int j=skewEngine->first[i];j<skewEngine->last[i];j++) {
            skewEngine->skewOutput[k + j] = lineRadon(i,j,skewEngine);
        }}
}



void showResultsRADN(float escala, float shift=0) {

if(saveSampleData){
    std::string dir = O_DIR;
    std::ofstream punto1i(dir + "punto1i.txt");
    std::ofstream punto2i(dir + "punto2i.txt");
    std::ofstream punto3i(dir + "punto3i.txt");
    std::ofstream punto1l(dir + "punto1l.txt");
    std::ofstream punto2l(dir + "punto2l.txt");
    std::ofstream punto3l(dir + "punto3l.txt");
    std::ofstream punto1c(dir + "punto1c.txt");
    std::ofstream punto2c(dir + "punto2c.txt");
    std::ofstream punto3c(dir + "punto3c.txt");

    for (int i = 0; i < 180; i++) {
        for (int j = 0; j < winSize / 2; j++) {
            punto1i << sampleData1[i][0][winSize / 2 + j] << " ";
            punto2i << sampleData1[i][1][winSize / 2 + j] << " ";
            punto3i << sampleData1[i][2][winSize / 2 + j] << " ";
        }
        punto1i << std::endl;
        punto2i << std::endl;
        punto3i << std::endl;
    }
    for (int i = 0; i < 180; i++) {
        for (int j = 0; j < winSize / 2; j++) {
            punto1i << sampleData1[i][0][winSize / 2 - 1 - j] << " ";
            punto2i << sampleData1[i][1][winSize / 2 - 1 - j] << " ";
            punto3i << sampleData1[i][2][winSize / 2 - 1 - j] << " ";
        }
        punto1i << std::endl;
        punto2i << std::endl;
        punto3i << std::endl;
    }

    // Escribir segundo bloque en punto1_k1.txt
    for (int i = 0; i < 180; i++) {
        for (int j = 0; j < winSize / 2; j++) {
            punto1l << sampleData2[i][0][j] << " ";
            punto2l << sampleData2[i][1][j] << " ";
            punto3l << sampleData2[i][2][j] << " ";
        }
        punto1l << std::endl;
        punto2l << std::endl;
        punto3l << std::endl;
    }
    for (int i = 0; i < 180; i++) {
        for (int j = 0; j < winSize / 2; j++) {
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
        for (int j = 0; j < winSize / 4; j++) {
            punto1c << sampleData3[i][0][j] << " ";
            punto2c << sampleData3[i][1][j] << " ";
            punto3c << sampleData3[i][2][j] << " ";
        }
        punto1c << std::endl;
        punto2c << std::endl;
        punto3c << std::endl;
    }
    for (int i = 0; i < 180; i++) {
        for (int j = 0; j < winSize / 4; j++) {
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
}

    std::vector<unsigned char> grey;
    std::vector<unsigned char> test;
    for(int i=0;i<dim;i++) {
        grey.push_back(pixels[i]/3);
        rgbColor c;
        if(skewAlgorithm==2||skewAlgorithm==3) c={
                                (unsigned char)((int)(outD[i]-shift)*escala),
                                (unsigned char)((int)(outD[i]-shift)*escala),
                                (unsigned char)((int)(outD[i]-shift)*escala)};
        test.push_back(c.R);
        test.push_back(c.G);
        test.push_back(c.B);
        }
    std::string  fn=O_DIR;
            ;
    lodepng::encode((fn+ "salida1.png").c_str(), grey, dimx, dimy, LCT_GREY);
    lodepng::encode((fn+ "salida2.png").c_str(), test, dimx, dimy, LCT_RGB);


}


#endif