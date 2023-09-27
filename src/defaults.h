/**
 * @file
 * @author Felipe Romero
 * @brief Constants and defines  (included in global.h)
 * @date 2022-11-17
 */

#ifndef DEFAULTS_H
#define DEFAULTS_H

#include <cmath>
#include "third-party/Lodepng.h"

#define IS_BIG_ENDIAN (*(uint16_t *)"\0\xff" < 0x100)
#define N2T(i) ((dimx * ((i) % dimx)) + ((i) / dimx))

#ifdef WIN32
#include <filesystem>
namespace fs = std::filesystem;
#define I_DIR "d:/input/"
#define O_DIR "d:/output/"
#define OUTPUT "./output"
#define INPUTDIR "./input"
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#define I_DIR "/users/felipe/input/"
#define O_DIR "/users/felipe/output/"
#define OUTPUT "./output"
#define INPUTDIR "./input"
#endif



#endif