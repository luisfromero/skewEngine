//
// Created by Felipe on 18-11-22.
//

#ifndef TVSSDEM_DRAWING_H
#define TVSSDEM_DRAWING_H

#endif //TVSSDEM_DRAWING_H


//#include <stdio.h>
//#include <stdlib.h>
#include <iostream>
//#include <math.h>
#include <iostream>
#include <iomanip>      	// std::setprecision
#include <fstream>              // std::ofstream
//#include <string>
#include <omp.h>
#include <climits>
#include <sstream>

//#include "defaults.h"
#include "auxf.h"
//#include "color.h"
//#include "Lodepng.h"
//#include "Global.h"
//#include "ViewshedPOV.h"
//#include "ViewshedMultiplePOV.h"

//extern char filename[100];
//extern char base_name[100];
//extern char maskname[100];
//extern char listname[100];
//extern char volcname[100];
//extern char horzname[100];
//extern char cvrgfname[100];
//extern char surname[100], volname[100];
//extern char surlname[100], vollname[100];
//extern bool towernotgiven;

std::vector<unsigned char> prep_image2D(float **data, float maxval, int n);
std::vector<unsigned char> prep_image1D(float *data, float maxval, int n);
std::vector<unsigned char> prep_image1D_Half(int *data, float maxval, int n);
std::vector<unsigned char> prep_image1D_HalfRotated(int *data, float maxval, int n);

void drawHalfMap(int ang, int *totalVS, int POV_i, int POV_j);
void drawDEM();
void drawTotalViewshedSinglePOVint(int ang, std::vector<int> totalVS, int POV_i, int POV_j);
void drawTotalViewshedDEM(int ang, float **totalVS);

void valuesToFloat(float *floatVS, float **totalVS);
void drawSectorSkewedDEM(int i);
void drawSectorViewshedSkewedDEM(int i, float **rotatedVS);
void drawSectorViewshedDEM(int i, float **sectorVS);


void v3_drawSkewDem(int n, int dimx,int dimy,float *data);