//
// Created by Felipe on 18-11-22.
//

#include "drawing.h"
using namespace std;


std::vector<unsigned char> prep_image2D(float **data, float maxval, int n) {

    my_palette = build_palette();
    std::vector<unsigned char> image;
    image.resize(dimx * n * dimy * 4);
    // if (verbose) printf("Preparando imagen \n");

    for (int i = 0; i < n * dimy; i++)
        for (int j = 0; j < dimx; j++) {
            int k = dimx * i + j;
            int v;
            v = 1024.0 * (data[i][j] / maxval);
            if (v < 0) v = 0;
            if (v > 1023) v = 1023;
            image[4 * k + 0] = my_palette[v].R;
            image[4 * k + 1] = my_palette[v].G;
            image[4 * k + 2] = my_palette[v].B;
            image[4 * k + 3] = 255;
        }

    return image;
}
std::vector<unsigned char> prep_image1D_Half(int *data, float maxval, int n) {

    my_palette = build_palette();
    std::vector<unsigned char> image;
    image.resize((dimx / 2) * n * (dimy / 2 + 200) * 4);
    // if (verbose) printf("Preparando imagen \n");

    for (int i = 0; i < n * dimy / 2 + 200; i++)
        for (int j = 0; j < dimx / 2; j++) {
            int k = (dimx / 2)* i + j;
            int v;
            v = 1024.0 * (data[(i + (dimy / 2 - 200)) * dimy + j] / maxval);
            if (v < 0) v = 0;
            if (v > 1023) v = 1023;
            image[4 * k + 0] = my_palette[v].R;
            image[4 * k + 1] = my_palette[v].G;
            image[4 * k + 2] = my_palette[v].B;
            image[4 * k + 3] = 255;
        }

    return image;
}
std::vector<unsigned char> prep_image1D_HalfRotated(int *data, float maxval, int n) {

    my_palette = build_palette();
    std::vector<unsigned char> image;
    image.resize((dimx / 2) * n * (dimy / 2 + 200) * 4);
    // if (verbose) printf("Preparando imagen \n");

    for (int i = 0; i < n * dimx / 2; i++)
        for (int j = 0; j < dimy / 2 + 200; j++) {
            int k = (dimy / 2 + 200) * i + j;
            int v;
            v = 1024.0 * (data[(dimy - j) * dimy + i] / maxval);
            if (v < 0) v = 0;
            if (v > 1023) v = 1023;
            image[4 * k + 0] = my_palette[v].R;
            image[4 * k + 1] = my_palette[v].G;
            image[4 * k + 2] = my_palette[v].B;
            image[4 * k + 3] = 255;
        }

    return image;
}
std::vector<unsigned char> prep_image1D(float *data, float maxval, int n) {

    my_palette = build_palette();
    std::vector<unsigned char> image;
    image.resize(dimx * n * dimy * 4);
    // if (verbose) printf("Preparando imagen \n");

    for (int i = 0; i < n * dimy; i++)
        for (int j = 0; j < dimx; j++) {
            int k = dimx * i + j;
            int v;
            v = 1024.0 * (data[i * dimy + j] / maxval);
            if (v < 0) v = 0;
            if (v > 1023) v = 1023;
            image[4 * k + 0] = my_palette[v].R;
            image[4 * k + 1] = my_palette[v].G;
            image[4 * k + 2] = my_palette[v].B;
            image[4 * k + 3] = 255;
        }

    return image;
}



void drawSectorSkewedDEM(int i, float **resultado2){

    // Draw sDEM
    if (mododibujo) {
        int n = 2;
        if (verbose) std::cout << "Creating a digital image of skewed DEM obtained on CPU: " << i << " degrees" << std::endl;
        std::vector<unsigned char> image = prep_image2D(resultado2, 200, n);
        char fn[100];
        sprintf(fn, "output/sDEM_sec%d_CPU.png", i);
        unsigned error = lodepng::encode(fn, image, dimx, n * dimy);
        if (error) std::cout << "Encoding error " << error << ": "<< lodepng_error_text(error) << std::endl;
        if (verbose) std::cout << "Digital image successfully created" << std::endl;
    }
}



void drawSectorViewshedDEM(int i, float **sectorVS) {

    // Draw normal sector viewshed on non-rotated map
    if (mododibujo) {
        float maxValue = 0.0;
        int n = 1;
        for (int y = 0; y < n * dimy; y++)
            for (int x = 0; x < dimx; x++)
                if (sectorVS[y][x] > maxValue)
                    maxValue = sectorVS[y][x];
        if (verbose) std::cout << "Creating a digital image of the sector viewshed on DEM obtained on CPU" << std::endl;
        std::vector<unsigned char> image = prep_image2D(sectorVS, (int)maxValue, n);
        char fn[100];
        int folder = 7;         // Al generar imágenes, utiliza carpetas consecutivas
        sprintf(fn, "output/VS_sec%d_CPU.png", i);
        unsigned error = lodepng::encode(fn, image, dimx, n * dimy);
        if (error) std::cout << "Encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
        if (verbose) std::cout << "Digital image successfully created" << std::endl;
    }
}

void drawSectorViewshedSkewedDEM(int i, float **rotatedVS) {

    // Draw sector viewshed on rotated map
    if (mododibujo) {
        float maxValue = 0.0;
        int n = 2;
        for (int y = 0; y < n * dimy; y++)
            for (int x = 0; x < dimx; x++)
                if (rotatedVS[y][x] > maxValue)
                    maxValue = rotatedVS[y][x];
        if (verbose) std::cout << "Creating a digital image of sector viewshed on skewed DEM obtained on CPU" << std::endl;
        std::vector<unsigned char> image = prep_image2D(rotatedVS, (int)maxValue, n);
        char fn[100];
        int folder = 7;         // Al generar imágenes, utiliza carpetas consecutivas
        sprintf(fn, "output/sVS_sec%d_CPU.png", i);
        unsigned error = lodepng::encode(fn, image, dimx, n * dimy);
        if (error) std::cout << "Encoding error " << error << ": "<< lodepng_error_text(error) << std::endl;
        if (verbose) std::cout << "Digital image successfully created" << std::endl;
    }
}


void valuesToFloat(float *floatVS, float **totalVS) {

    // Values to float
    if (modofloat) {
        for (int y = 0; y < dimy; y++)
            for (int x = 0; x < dimx; x++)
                floatVS[y * dimx + x] = totalVS[y][x]*surScale;
        char fn[100];
        bool transpose = false;
        sprintf(fn, "resultado", transpose);
        saveFloats(floatVS, fn, transpose);
        transpose = true;
        //sprintf(fn, "viewshed_acc_transpose%d_CPU", transpose);
        //saveFloats(floatVS, fn, transpose);
    }
}

void drawSectorSkewedDEM(int i) {

    // Draw sDEM
    if (mododibujo) {
        int n = 2;
        if (verbose) std::cout << "Creating a digital image of skewed DEM obtained on CPU: " << i << " degrees" << std::endl;
        std::vector<unsigned char> image = prep_image2D(resultado, 200, n);
        char fn[100];
        sprintf(fn, "output/sDEM_sec%d_CPU.png", i);
        unsigned error = lodepng::encode(fn, image, dimx, n * dimy);
        if (error) std::cout << "Encoding error " << error << ": "<< lodepng_error_text(error) << std::endl;
        if (verbose) std::cout << "Digital image successfully created" << std::endl;
    }
}


void drawTotalViewshedDEM(int ang, float **totalVS) {

    // Draw normal total viewshed on non-rotated map
    if (mododibujo) {
        float maxValue = 0.0;
        int n = 1;
        for (int y = 0; y < n * dimy; y++)
            for (int x = 0; x < dimx; x++)
                if (totalVS[y][x] > maxValue)
                    maxValue = totalVS[y][x];
        if (verbose) std::cout << "Creating a digital image of the total viewshed on DEM obtained on CPU -> maxValue: " << maxValue << std::endl;
        std::vector<unsigned char> image = prep_image2D(totalVS, (int)maxValue, n);
        char fn[100];
        int folder = 7;         // Al generar imágenes, utiliza carpetas consecutivas
        sprintf(fn, "output/TotalVS%d_acc%d_CPU.png", dimx, ang);
        unsigned error = lodepng::encode(fn, image, dimx, n * dimy);
        if (error) std::cout << "Encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
        if (verbose) std::cout << "Digital image successfully created" << std::endl;
    }
}

void drawHalfMap(int ang, int *totalVS, int POV_i, int POV_j) {

    // Draw normal total viewshed on non-rotated map
    if (mododibujo) {
        float maxValue = 0.0;
        int n = 1;
        for (int y = 0; y < n * dimy; y++)
            for (int x = 0; x < dimx; x++)
                if (totalVS[y * dimx + x] > maxValue)
                    maxValue = totalVS[y * dimx + x];
        if (verbose) std::cout << "Creating a digital image of the total viewshed on DEM obtained on CPU" << std::endl;
        std::vector<unsigned char> image = prep_image1D_Half(totalVS, (int)maxValue, n);
        char fn[100];
        int folder = 7;         // Al generar imágenes, utiliza carpetas consecutivas
        sprintf(fn, "output/VS%d_max(%d-%d)POV%d_CPU.png", dimx/2 + 200, POV_i, POV_j, ang);
        unsigned error = lodepng::encode(fn, image, dimx/2, n * dimy / 2 + 200);
        if (error) std::cout << "Encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
        if (verbose) std::cout << "Digital image successfully created" << std::endl;
    }
}

void drawDEM() {

    // Draw sDEM
    std::cout << "Dimension: " << dimx << " " << dimy << std::endl;

    if (mododibujo) {
        int n = 1;
        float *a = new float[dimy*dimx];
        for (int y = 0; y < n * dimy; y++)
            for (int x = 0; x < dimx; x++)
                a[y * dimy + x] = (float)heights[y * dimx + x];
        if (verbose) std::cout << "Creating a digital image of the DEM obtained on CPU" << std::endl;
        std::vector<unsigned char> image = prep_image1D(a, 200, n);
        char fn[100];
        sprintf(fn, "output/DEM_CPU.png");
        unsigned error = lodepng::encode(fn, image, dimx, n * dimy);
        if (error) std::cout << "Encoding error " << error << ": "<< lodepng_error_text(error) << std::endl;
        if (verbose) std::cout << "Digital image successfully created" << std::endl;
        delete[] a;
    }
}

void drawTotalViewshedSinglePOVint(int ang, std::vector<int> totalVS, int POV_i, int POV_j) {

    // Draw normal total viewshed on non-rotated map
    if (mododibujo) {
        float maxValue = 0.0;
        float *a = new float[dimy*dimx];
        int n = 1;
        for (int y = 0; y < n * dimy; y++)
            for (int x = 0; x < dimx; x++) {
                a[y * dimx + x] = (int)totalVS[y * dimx + x];
                if (totalVS[y * dimx + x] > maxValue)
                    maxValue = totalVS[y * dimx + x];
            }
        // if (verbose) std::cout << "Creating a digital image of the total viewshed on DEM obtained on CPU" << std::endl;
        std::vector<unsigned char> image = prep_image1D(a, (int)maxValue, n);
        char fn[100];
        int folder = 7;         // Al generar imágenes, utiliza carpetas consecutivas
        sprintf(fn, "output/VS%d_max(%d-%d)POV%d_CPU.png", dimx, POV_i, POV_j, ang);
        unsigned error = lodepng::encode(fn, image, dimx, n * dimy);
        if (error) std::cout << "Encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
        if (verbose) std::cout << "Digital image successfully created" << std::endl;
        delete[] a;
    }
}



void drawSkewedMask(int ang, bool *sMask, int POV_i, int POV_j) {

    // Draw normal total viewshed on non-rotated map
    if (mododibujo) {
        //float maxValue = 0.0;
        int n = 2;
        float *a = new float[n * dimy * dimx];
        for (int y = 0; y < n * dimy; y++)
            for (int x = 0; x < dimx; x++) {
                a[y * dimx + x] = (int)sMask[y * dimx + x];
                //if (sMask[y * dimx + x] > maxValue)maxValue = sMask[y * dimx + x];
            }
        if (verbose) std::cout << "Creating a digital image of the total viewshed on DEM obtained on CPU" << std::endl;
        std::vector<unsigned char> image = prep_image1D(a,1, n);
        char fn[100];
        int folder = 7;         // Al generar imágenes, utiliza carpetas consecutivas
        sprintf(fn, "output/sMASK%d(%d-%d)POV%d_CPU.png", dimx, POV_i, POV_j, ang);
        unsigned error = lodepng::encode(fn, image, dimx, n * dimy);
        if (error) std::cout << "Encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
        if (verbose) std::cout << "Digital image successfully created" << std::endl;
        delete[] a;
    }
}

void v2drawTotalViewshedSinglePOV(int ang, float *fulltotalVS) {

    int id = omp_get_thread_num();
    float *totalVS = new float[dimy * dimx];

    // Draw normal total viewshed on non-rotated map
    if (mododibujo) {
        float maxValue = 0.0;
        int n = 1;
        for (int y = 0; y < n * dimy; y++)
            for (int x = 0; x < dimx; x++) {
                totalVS[y * dimy + x] = fulltotalVS[id * dimy * dimx + y * dimy + x];
                if (totalVS[y * dimx + x] > maxValue)
                    maxValue = totalVS[y * dimx + x];
            }

        if (verbose) std::cout << "Creating a digital image of the total viewshed on DEM obtained on CPU" << std::endl;
        std::vector<unsigned char> image = prep_image1D(totalVS, (int)maxValue, n);
        char fn[100];
        int folder = 7;         // Al generar imágenes, utiliza carpetas consecutivas
        sprintf(fn, "output/VS%d_max(%d-%d)POV%d_CPU.png", dimx, 0, 0, ang);
        unsigned error = lodepng::encode(fn, image, dimx, n * dimy);
        if (error) std::cout << "Encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
        if (verbose) std::cout << "Digital image successfully created" << std::endl;
    }

    delete[] totalVS;
}

void drawTotalViewshedSinglePOV(int ang, float *totalVS, int POV_i, int POV_j) {

    // Draw normal total viewshed on non-rotated map
    if (mododibujo) {
        float maxValue = 0.0;
        int n = 1;
        for (int y = 0; y < n * dimy; y++)
            for (int x = 0; x < dimx; x++)
                if (totalVS[y * dimx + x] > maxValue)
                    maxValue = totalVS[y * dimx + x];
        if (verbose) std::cout << "Creating a digital image of the total viewshed on DEM obtained on CPU" << std::endl;
        std::vector<unsigned char> image = prep_image1D(totalVS, (int)maxValue, n);
        char fn[100];
        int folder = 7;         // Al generar imágenes, utiliza carpetas consecutivas
        sprintf(fn, "output/VS%d_max(%d-%d)POV%d_CPU.png", dimx, POV_i, POV_j, ang);
        unsigned error = lodepng::encode(fn, image, dimx, n * dimy);
        if (error) std::cout << "Encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
        if (verbose) std::cout << "Digital image successfully created" << std::endl;
    }
}


void drawHalfMapRotated(int ang, int *totalVS, int POV_i, int POV_j) {
    // Draw normal total viewshed on non-rotated map
    if (mododibujo) {
        float maxValue = 0.0;
        int n = 1;
        for (int y = 0; y < n * dimy; y++)
            for (int x = 0; x < dimx; x++)
                if (totalVS[y * dimx + x] > maxValue)
                    maxValue = totalVS[y * dimx + x];
        if (verbose) std::cout << "Creating a digital image of the total viewshed on DEM obtained on CPU" << std::endl;
        std::vector<unsigned char> image = prep_image1D_HalfRotated(totalVS, (int)maxValue, n);
        char fn[100];
        int folder = 7;         // Al generar imágenes, utiliza carpetas consecutivas
        sprintf(fn, "output/multi-SVS%d_rotated_drone_path.png", dimx/2 + 200);
        unsigned error = lodepng::encode(fn, image, dimx/2 + 200, n * dimy / 2);
        if (error) std::cout << "Encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
        if (verbose) std::cout << "Digital image successfully created" << std::endl;
    }
}


void v3_drawSkewDem(int n, int dimx,int dimy,float *data)
{
    char fn[100];
    int folder = 7;         // Al generar imágenes, utiliza carpetas consecutivas
    std::vector<unsigned char> image;
    sprintf(fn, "output/v3/%03d.png",n);

    my_palette = build_palette();
    image.resize(dimx *dimy*4);
    // if (verbose) printf("Preparando imagen \n");

    for (int i = 0; i < dimy; i++)
        for (int j = 0; j < dimx; j++) {
            int k = dimx * i + j;
            int v;
            v = 1024.0 * (data[i * dimx + j] / 192.0);  //Torrecilla mide 1900
            if (v < 0) v = 0;
            if (v > 1023) v = 1023;
            image[4 * k + 0] = my_palette[v].R;
            image[4 * k + 1] = my_palette[v].G;
            image[4 * k + 2] = my_palette[v].B;
            image[4 * k + 3] = 255;
        }

    lodepng::encode(fn, image, dimx,dimy);

}