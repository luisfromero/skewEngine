//
// Created by Felipe on 17-11-22.
//
#ifndef DEFAULTS_H
#define DEFAULTS_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include "Lodepng.h"
//#include <string>
#define WINPAUSE system("pause")


#ifndef TVSSDEM_DEFAULTS_H
#define TVSSDEM_DEFAULTS_H
#endif //TVSSDEM_DEFAULTS_H

#define IS_BIG_ENDIAN (*(uint16_t *)"\0\xff" < 0x100)
#define STEP 10
#define UTMN 4070000
#define UTME 0310000
#define CACHED "../cached"
#define OUTPUT "./output"
#define INPUTDIR "./input"

// UTM zone
#define ZONE "30S"

// obs_h=10;       observer's height in meters
#define OBS_H 1.5

//T fila es i%dimy
//T colu es i/dimy
#define T2N(i) ((dimy * ((i) % dimy)) + ((i) / dimy))

//N fila es i/dimx
//N colu es i%dimx
#define N2T(i) ((dimx * ((i) % dimx)) + ((i) / dimx))



const double PI = M_PI;
const double mpi = M_PI / 2;
const double tmpi = 3 * M_PI / 2;
const double tograd = 360 / (2 * M_PI); // conver const of radians to degree
const double torad = (2 * M_PI) / 360;  // conver const of degree to radians
const float PI_F = 3.14159265358979f;
const int isolationindex = 2; // 0 max 1 m_arit 2 m_harm 3 m_geom

#endif