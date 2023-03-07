//
// Created by Felipe on 20-2-23.
//

#ifndef SKEWENGINE_HELPER_H
#define SKEWENGINE_HELPER_H
#include "global.h"
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <sys/stat.h>
#include <limits>

#ifdef WIN32
#include <winsock2.h>
#include <windows.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <cfloat>
#endif

namespace helper
{


    /** Cooley-Tukey FFT (in-place, breadth-first, decimation-in-frequency)
 * Better optimized but less intuitive
 * !!! Warning : in some cases this code make result different from not optimased version above (need to fix bug)
 * The bug is now fixed @2017/05/30
 */
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

    void ifft(CArray& x)
    {
        x = x.apply(std::conj);
        fft( x );
        x = x.apply(std::conj);
        x /= x.size();
    }


    char * str_upr(char *s) {

        unsigned c;
        unsigned char *p = (unsigned char *)s;
        while (c = *p) *p++ = toupper(c);
        return s;
    }


    bool exists(const char * filename) {
        struct stat fileinfo;
        return !stat(filename, &fileinfo);
    }

    bool directoryExists(const char * filename) {
        struct stat fileinfo;
        return !stat(filename, &fileinfo);
    }

    long filesize(char *filename) {

        long size = 0;
        struct stat fileinfo;
        if (!stat(filename, &fileinfo))
            size = fileinfo.st_size;
        return size;
    }


#ifdef WIN32
    void create_directory(const char *name) {

        CreateDirectoryA(name,NULL);
    }
#else
    void create_directory(const char *name) {
    mkdir(name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}
#endif


// Lee un archivo PNG y lo convierte a un vector de pixeles en escala de grises
    void read_png(const std::string filename, std::vector<short> &pixels, unsigned &imgWidth, unsigned &imgHeight) {
        std::vector<unsigned char> image;

        // Lee el archivo PNG

        unsigned error = lodepng::decode(image, imgWidth, imgHeight, filename);
        if (error) {
            throw std::runtime_error(lodepng_error_text(error));
        }

        // Convierte el vector de bytes a un vector de pixeles RGB
        pixels.resize(imgWidth * imgHeight);
        for (unsigned y = 0; y < imgHeight; y++) {
            for (unsigned x = 0; x < imgWidth; x++) {
                unsigned i = y * imgWidth + x;
                pixels[i]=  image[i * 4 + 0] + image[i * 4 + 1] + image[i * 4 + 2];
            }
        }
        return;
    }



    pair_t getMinMax(float * datos)
    {
        pair_t resultado;
        resultado.min=FLT_MAX;
        resultado.max=FLT_MIN;
        for(int i=0;i<dimx*dimy;i++)
        {
            resultado.min=datos[i]<resultado.min?datos[i]:resultado.min;
            resultado.max=datos[i]>resultado.max?datos[i]:resultado.max;
        }
        return resultado;
    }

    pair_t getMinMax(double * datos)
    {
        pair_t resultado;
        resultado.min=DBL_MAX;
        resultado.max=DBL_MIN;
        for(int i=0;i<dimx*dimy;i++)
        {
            resultado.min=datos[i]<resultado.min?datos[i]:resultado.min;
            resultado.max=datos[i]>resultado.max?datos[i]:resultado.max;
        }
        return resultado;
    }

    pair_t getMinMax(float ** datos)
    {
        pair_t resultado;
        resultado.min=FLT_MAX;
        resultado.max=0;
        for(int i=0;i<dimy;i++)
            for(int j=0;j<dimx;j++)
            {
                resultado.min=datos[i][j]<resultado.min?datos[i][j]:resultado.min;
                resultado.max=datos[i][j]>resultado.max?datos[i][j]:resultado.max;
            }
        return resultado;
    }





}









#endif //SKEWENGINE_HELPER_H
