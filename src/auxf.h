//
// Created by Felipe on 17-11-22.
//

#ifndef TVSSDEM_AUX_H
#define TVSSDEM_AUX_H

#include "global.h"

char * str_upr(char *s);
bool exists(const char * filename);
long filesize(char *filename);
void create_directory(const char *name);

double * readHeights(char *x, bool floats, bool sameendian);
double * readHeights(char *x);
void readHeights(char *nmde, float *&h_DEM);
void saveFloats(float * c, char *fname, bool transpose);
header_t parse_hdr(char *filename);
pair_t getMinMax(double * data);
pair_t getMinMax(float * data);
pair_t getMinMax(float ** data);
bool check_file_v2(char *argv);
void fillin(float *&h_DEM);


/// <summary>
/// Averigua el tamaño del archivo: si es flt guarda 4 bytes de elevacion; si es bil, suele guardar 2 bytes
/// </summary>

void menu();

void configure_v2(int argc, char *argv[]);
void check_args_v2(int argc, char **argv);


/// <summary>
/// Reserva de memoria para las elevaciones, sin remapear
/// Reserva la memoria para el mapa remapeado
/// Trabajamos entre 0 y 90º y nos permite calcular cuencas 0-90 180-270
/// Para el resto, calcularíamos la traspuesta
/// En la versión sencilla, necesitamos el doble de espacio
/// Asigna el valor inicial (angulo sector 0)
/// </summary>
void allocate();

/// <summary>
/// Libera la memoria reservada por allocate()
/// </summary>
void deallocate();

void allocate_v3();
void allocateV3(int skewAlgorithm=0);
void deallocateV3(int skewAlgorithm=0);

//CPU sector parallel

void allocThreadSectorData(float **&skwDEM, float **&rotatedVS);
void resetThreadSectorData(float **skwDEM, float **rotatedVS);
void deallocThreadSectorData(float **skwDEM, float **rotatedVS, int *destiny, float * ratio,float *sectorVS);

void allocSharedData(float **totalVS);
void resetSharedData(float **totalVS,float *floatVS);
void deallocSharedData(float **totalVS);

template <typename T>
void readBilRaster(char *inputfilename, T *&h_DEM)
{
    //simple non-square input test:
    //  dimy=1080;dimx=1920;

    FILE *f;
    f = fopen(inputfilename, "rb");
    if (f == NULL) {
        printf("Error opening %s\n", inputfilename);
    }
    else {
        for (int i = 0; i < dimy; i++) {
            for (int j = 0; j < dimx; j++) {
                short num;
                fread(&num, 2, 1, f);
                h_DEM[dimx * i + j] = ((T) num) / step; //internal representation from top to bottom (inner loop)
            }
            //simple non-square input test:
            //fseek(f, 2*(2000-dimx), SEEK_CUR);
        }
        fclose(f);
        pair_t mm= getMinMax(h_DEM);
        printf("Input model readed, with extreme values (/step): %5.1f - %6.1f\n",mm.min*step,mm.max*step);
    }
}



void read_png(const std::string filename, std::vector<short> &pixels, unsigned &imgWidth, unsigned &imgHeight);




#endif //TVSSDEM_AUX_H

