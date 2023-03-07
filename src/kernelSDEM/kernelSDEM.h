/**
 * @file kernelSDEM.h
 * @author Felipe Romero
 * @brief Funciones específicas del kernel para el módulo de cálculo de total viewshed
 */

#ifndef KSDEM_H
#define KSDEM_H

//Defaults
#define STEP 10
#define OBS_H 1.5


double obsheight=OBS_H; // Elevación en metros
float step=STEP; // Grid size
float POVh=OBS_H; // Elevación en scala step
float surScale; // Escala en superficie
int utm_n,utm_e; // Coordenadas UTM
bool floatsDEM=false; // Tipo de datos. Por ahora, short y float


/**
 * Lee datos de un archivo y (enteros, flotantes, double, ...)
 * @tparam T
 * @param inputfilename Nombre del archivo
 * @param inputData Destino de los datos leídos
 * @param crop_x Aplicar recorte en x respecto a los datos originales
 * @param crop_y Aplicar recorte en x respecto a los datos originales
 */
template <typename T>
void readBilData(char *inputfilename, T *&inputData, int crop_x=0, int crop_y=0)
{
    int sdimx=dimx;
    dimx=dimx-crop_x;
    dimy=dimy-crop_y;
    FILE *f;
    f = fopen(inputfilename, "rb");
    if (f == NULL) {
        printf("Error opening %s\n", inputfilename);
    }
    else {
        for (int i = 0; i < dimy; i++) {
            for (int j = 0; j < dimx; j++) {
                short num;
                fread(&num, sizeof(short), 1, f);
                inputData[dimx * i + j] = ((T) num) / 10.0; //internal representation from top to bottom (inner loop)
            }
            fseek(f, sizeof(short)*(sdimx-dimx), SEEK_CUR);
        }
        fclose(f);
        pair_t mm= helper::getMinMax(inputData);
        if(verbose)printf("Succesfully read DEM, with extreme values (/step): %5.1f - %6.1f\n",mm.min*step,mm.max*step);
    }
}


header_t parse_hdr(char *filename){

    header_t hdr;
    char tmpfn[100];
    char hdrfile[100];
    hdr.nitems = 0;

    strcpy(tmpfn, filename);
    char *prefix = strtok(tmpfn, ".");
    strcpy(hdrfile, prefix);
    strcat(hdrfile, ".hdr");
    if (!helper::exists(hdrfile)) {
        printf("Missing header file %s. Using default cell size %d and UTM coordinates .\n", hdrfile, 10);
        return hdr;
    }
    FILE *file = fopen(hdrfile, "r");
    char buffer[200];
    char *name, *value;
    char names[200];
    // char value[200];
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        name = strtok (buffer, " ,.-=:");
        helper::str_upr(name);
        value = strtok(NULL, "\r\n");

        char *tmp;
        tmp = (char *)malloc(strlen(name) + 1);
        strcpy(tmp, name);
        hdr.keys[hdr.nitems] = tmp;
        tmp = (char *)malloc(strlen(value) + 1);
        strcpy(tmp, value);
        hdr.values[hdr.nitems] = tmp;
        hdr.nitems++;
        // printf("%s %s\n", name, value);
    }
    return hdr;
}



bool check_dem(std::string argv) {
    std::string name;
    name=argv;
    std::string filename=argv;
    std::string rawname = name.substr(0, name.find_last_of("."));
    std::string hdrname=rawname+".hdr";

    header_t hdr;
    char cadena[100];
    char *extension;
    char *basen;
    long size;



    hdr.nitems = 0;
    int i = sscanf(argv.c_str(), "%d_%d_%f.bil", &utm_n, &utm_e, &step);
    int hdimx = 0, hdimy = 0, hstep = 10, hutmn = 0, hutme = 0;
    if (i != 4) {
        // Este archivo no es nuestro
        hdr = parse_hdr((char *)hdrname.c_str());
        if (hdr.nitems == 0) {
            for (int i = 0; i < hdr.nitems; i++) {
                if (!strcmp(hdr.keys[i], "NROWS")) sscanf(hdr.values[i], "%d", &hdimy);
                if (!strcmp(hdr.keys[i], "NCOLS")) sscanf(hdr.values[i], "%d", &hdimx);
                if (!strcmp(hdr.keys[i], "CELLSIZE")) sscanf(hdr.values[i], "%d", &hstep);
                if (!strcmp(hdr.keys[i], "XLLCORNER")) sscanf(hdr.values[i], "%d", &hutme);
                if (!strcmp(hdr.keys[i], "YLLCORNER")) sscanf(hdr.values[i], "%d", &hutmn);
            }
            utm_e = hutme;
            utm_n = hutmn + hdimy * hstep;
            step = hstep;
            printf("%d %d %f\n", utm_e, utm_n, step);
        }

    }
    size = helper::filesize((char *)filename.c_str());
    if (size > 0)
        dimx = dimy = (int)sqrt((double)(size / (floatsDEM ? 4 : 2)));
    else
        return false;
    if ((hdr.nitems != 0) && ((dimx != hdimx) || (dimy != hdimy))) return false;
    N = dim = dimx * dimy;
    return true;
}



/**
 *
 * @param filename
 */
void configureSDEM(char *filename)
{
    char fn[100];
    strcpy(fn,I_DIR);
    strcat(fn,filename);
    inputD = new float[dim];
    readBilData(fn, inputD);
    surScale=M_PI/(360*step*step);
    POVh=obsheight/step;
    if(verbose) {
        printf("Allocating DEM (from %s, with filesize: %dx%d and step %f) and setting observer's height to %f\n", filename, dimx, dimy,step,obsheight);
    }
    check_dem(fn);
}


void kernelV3(skewEngine<float> *skewer)
{
    for (int i = 0; i < skewer->skewHeight; i++) {
        float r =pow(skewer->scale,2);
        int inicio= skewer->first[i];
        int final = skewer->last[i];

        float cv,  max_angle, open_delta_d, delta_d, cAngle;
        float h;
        bool visible, above, opening, closing;
        int rowptr= skewer->skewWidth * i;

        for (int j = inicio; j < final; j++) {

            cv = 0;
            h = skewer->skewInput[rowptr + j]+POVh;

            max_angle = -2000;
            visible = true;
            open_delta_d = 0;

            for (int k = j + 1; k >= inicio && k < final; k++) {
                delta_d = k - j;
                cAngle = (skewer->skewInput[rowptr + k] - h) / delta_d;
                above = (cAngle > max_angle);
                opening = above && (!visible);
                closing = (!above) && (visible);
                visible = above;
                max_angle = std::max(cAngle, max_angle);
                if (opening) open_delta_d = delta_d;
                if (closing) cv += (delta_d * delta_d - open_delta_d * open_delta_d);
            }

            max_angle = -2000;
            visible = true;
            open_delta_d = 0;

            for (int k = j - 1; k >= inicio && k < final; k--) {
                delta_d = j - k;
                //float idelta=_mm_cvtss_f32(_mm_rcp_ss( _mm_set_ss((int)delta_d) ));
                //cAngle = (skwDEM[i][k] - h) * idelta ;
                cAngle = (skewer->skewInput[rowptr + k] - h) /delta_d;
                above = (cAngle > max_angle);
                opening = above && (!visible);
                closing = (!above) && (visible);
                visible = above;
                max_angle = std::max(cAngle, max_angle);
                if (opening) open_delta_d = delta_d;
                if (closing) cv += (delta_d * delta_d - open_delta_d * open_delta_d);
            }
            skewer->skewOutput[rowptr + j] = cv * r;
        }
    }

}
__global__
void kernelV3cuda(float *skewOutput, float *skewInput, int dim_i, int skewHeight, unsigned short *d_last, unsigned short *d_first, float POVh, int angle) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= skewHeight)return;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int start, end;
    start=d_first[row];
    end=d_last[row];
    if(col>=end&&col<start)return;
    float torads=M_PI/180;
    float inrads=angle*torads;
    float r = 1.0 / pow(cos(inrads),2);


    float cv, h, max_angle, open_delta_d, delta_d, cAngle;
    bool visible, above, opening, closing;

//    if (row < endGPUbatch && col >= start && col < end)
    {
        cv = 0;                                                                                 // Ya tengo observador en j
        h = skewInput[row * dim_i + col] + POVh;
        max_angle = -2000;
        visible = true;
        open_delta_d = 0;

        for (int k = col + 1; k >= start && k < end; k++) {                                       // Calculo cv a dcha

            delta_d = k - col;
            cAngle = (skewInput[row * dim_i + k] - h) / delta_d;
            above = (cAngle > max_angle);
            opening = above && (!visible);
            closing = (!above) && (visible);

            visible = above;
            max_angle = std::max(cAngle, max_angle);
            if (opening) open_delta_d = delta_d;
            if (closing) cv += (delta_d * delta_d - open_delta_d * open_delta_d);
        }

        max_angle = -2000;
        visible = true;
        open_delta_d = 0;

        for (int k = col - 1; k >= start && k < end; k--) {                                      // Calculo cv a izqda

            delta_d = col - k;
            cAngle = (skewInput[row * dim_i + k] - h) / delta_d;
            above = (cAngle > max_angle);
            opening = above && (!visible);
            closing = (!above) && (visible);

            visible = above;
            max_angle = std::max(cAngle, max_angle);
            if (opening) open_delta_d = delta_d;
            if (closing) cv += (delta_d * delta_d - open_delta_d * open_delta_d);
        }

        skewOutput[row * dim_i + col] = cv * r;
    }

}




void showResultsSDEM()
{

}





#endif