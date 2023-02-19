//
// Created by Felipe on 17-11-22.
//
#ifndef DEFAULTS_H
#define DEFAULTS_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include "third-party/Lodepng.h"
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
#define N2T(i) ((dimx * ((i) % dimx)) + ((i) / dimx))



const double PI = M_PI;

#endif