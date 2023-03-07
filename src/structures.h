//
// Created by Felipe on 19-2-23.
//

#ifndef SKEWENGINE_STRUCTURES_H
#define SKEWENGINE_STRUCTURES_H

#include <complex>
#include <valarray>

struct color {unsigned char R,G,B;};
struct hsvColor {float H,S,V;};
struct rgbColor {unsigned char R,G,B;};

typedef struct  {
    float min;
    float max;
}    pair_t;

typedef struct  {
    int x;
    int y;
}    point_t;

/*
typedef struct  {
    int dimx;
    int dimy;
    int bw;
    double obs_h;
    double step;
} sectorgeometry_t ;
*/

typedef struct {
    int nitems;
    char * keys[50];
    char * values[50];
}header_t;

typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;


#endif //SKEWENGINE_STRUCTURES_H
