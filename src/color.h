/**
 * @file
 * @author Felipe Romero
 * @brief Funciones para la creación y manipulación de mapas de colores, datos HSV o RGB, etc.
 * @date 2022-11-17
 */

#ifndef SKE_COLOR_H
#define SKE_COLOR_H

#include "structures.h"

color * build_palette();
rgbColor HSVtoRGB(hsvColor hsv);
#endif //TVSSDEM_COLOR_H




