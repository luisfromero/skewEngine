//
// Created by Felipe on 17-11-22.
//
#define _USE_MATH_DEFINES
#include "auxf.h"
#include "defaults.h"

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <sys/stat.h>
//#include <float.h>
#include <limits>

#ifdef WIN32
#include <winsock2.h>
#include <windows.h>
#else
#include <sys/time.h>
//#include <boost/filesystem.hpp>
#include <sys/types.h>
#include <unistd.h>
#endif



using namespace std;

extern int N,dimx,dimy,dim;

char filename[100];     // Input filename
char base_name[100];    // Input filename without path and extension
char maskname[100];     // Mask file with boolean data
char listname[100];     // List of tower locations
char surname[100];      // Total viewshed result
char surlname[100];      // Total viewshed result (log scale)
char pngname[100];     // Graphical output filename
char pnglname[100];     // Graphical output filename (log scale)
char cvrgfname[100];    // Coverage output
char idname[100];       // Track or tower identifier


extern float step;
extern int dim, utm_n, utm_e, dimx, dimy;
extern int mode, identifier, runmode;
extern bool verbose, silent, inputmask, volume, fullstore, showtime;
extern int tower, *towers, ntowers, cnttower;
extern double obsheight;
extern bool floatsDEM, endianOK;
bool towernotgiven = true;
extern int nthreads;
extern int deviceIndex;
extern bool mododibujo,modotiming;
extern float surScale;




#include "color.h"



//-----------------------------------------------------------------------------------------------------
//                    Compatibility functions
//-----------------------------------------------------------------------------------------------------

char * str_upr(char *s) {

    unsigned c;
    unsigned char *p = (unsigned char *)s;
    while (c = *p) *p++ = toupper(c);
    return s;
}

//-----------------------------------------------------------------------------------------------------
//                    File and directory functions
//-----------------------------------------------------------------------------------------------------

bool exists(const char * filename) {

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



//-----------------------------------------------------------------------------------------------------
//                    Raster functions
//-----------------------------------------------------------------------------------------------------



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


//-----------------------------------------------------------------------------------------------------
//                    Raster functions
//-----------------------------------------------------------------------------------------------------




//Si step=10, la unidad de elevación es el decámetro
double * readHeights(char *nmde, bool floats, bool sameendian) {

    //Read heights of the points on the map and positions according to order A
    FILE *f;
    double *h = new double[dim * sizeof(double)];
    f = fopen(nmde, "rb");
    if (f == NULL) {
        printf("Error opening %s\n", nmde);
    }
    else {
        if (sameendian && !floats)
            for (int i = 0; i < dim; i++) {
                short num;
                fread(&num, 2, 1, f);
                h[N2T(i)] = (1.0 * num) / step;  //internal representation from top to bottom (inner loop)
            }

        if (sameendian && floats)
            for (int i = 0; i < dim; i++) {
                float num;
                fread(&num, 4, 1, f);
                h[N2T(i)] = num / step;      //internal representation from top to bottom (inner loop)
            }
    }

    return h;
}

//Si step=10, la unidad de elevación es el decámetro
double * readHeights(char *nmde) {

    //Read heights of the points on the map and positions according to order A
    FILE *f;
    double *h = new double[dim * sizeof(double)];
    f = fopen(nmde, "rb");
    if (f == NULL) {
        printf("Error opening %s\n", nmde);
    }
    else {
        for (int i = 0; i < dim; i++) {
            short num;
            fread(&num, 2, 1, f);
            h[i] = (1.0 * num) / step; //internal representation from top to bottom (inner loop)
        }
    }
    return h;
}

//Si step=10, la unidad de elevación es el decámetro
void readHeights(char *nmde, float *&h_DEM) {

    //Read heights of the points on the map and positions according to order A
    FILE *f;
    f = fopen(nmde, "rb");
    if (f == NULL) {
        printf("Error opening %s\n", nmde);
    }
    else {
        for (int i = 0; i < dim; i++) {
            short num;
            fread(&num, 2, 1, f);
            h_DEM[i] = (1.0 * num) / step; //internal representation from top to bottom (inner loop)
        }
    }
}

void saveFloats(float * c, char *fname, bool transpose) {

    char ffname[100];
    sprintf(ffname, "%s/%s.flt", OUTPUT, fname);
    FILE *f;
    int n = 0;
    f = fopen(ffname, "wb");
    if (transpose) {
        for(int i = 0; i < dim; i++) {
            float x = c[N2T(i)];
            fwrite(&x, 1, 4, f);
        }
    }
    else
        fwrite(c, 4, dim, f);

    fclose(f);
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
    if (!exists(hdrfile)) {
        printf("Missing header file %s. Using default cell size %d and UTM coordinates .\n", hdrfile, STEP);
        return hdr;
    }
    FILE *file = fopen(hdrfile, "r");
    char buffer[200];
    char *name, *value;
    char names[200];
    // char value[200];
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        name = strtok (buffer, " ,.-=:");
        str_upr(name);
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

bool check_file_v2(char *argv) {

    header_t hdr;
    char cadena[100];
    char *extension;
    char *basen;
    long size;
    strcpy(base_name,argv);
    strtok(base_name,".");
    //strcpy(cadena,INPUTDIR);
    if (!exists(INPUTDIR)) create_directory(INPUTDIR);
    if (!exists(OUTPUT)) create_directory(OUTPUT);
    sprintf(surname, "%s_sur", base_name);
    sprintf(surlname, "%s_surlog", base_name);


    if (!silent) printf(filename, "%s/%s\n", INPUTDIR, argv);
    sprintf(filename, "%s/%s", INPUTDIR, argv);
    hdr.nitems = 0;
    int i = sscanf(argv, "%d_%d_%f.bil", &utm_n, &utm_e, &step);
    int hdimx = 0, hdimy = 0, hstep = 10, hutmn = 0, hutme = 0;
    if (i != 3) {
        // Este archivo no es nuestro
        hdr = parse_hdr(filename);
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
    if ((extension = strrchr(filename, '.')) != NULL ) floatsDEM = !strcmp(extension, ".flt");
    // strcpy(filename, cadena);
    size = filesize(filename);
    if (size > 0)
        dimx = dimy = (int)sqrt((double)(size / (floatsDEM ? 4 : 2)));
    else
        return false;
    if ((hdr.nitems != 0) && ((dimx != hdimx) || (dimy != hdimy))) return false;
    N = dim = dimx * dimy;
    return true;
}

void fillin(float *&h_DEM) {
    //readHeights(filename, h_DEM);
    //readBilRaster(filename, h_DEM);
    //for(int i=0;i<dim;i++)h_DEM[i]=heights[i];
}



//-----------------------------------------------------------------------------------------------------
//                    Menues and command line functions
//-----------------------------------------------------------------------------------------------------

void menu() {

    std::string ans;

    std::cout << "Available execution modes for sDEM:" << std::endl << std::endl;;

    std::cout << "  [0] -> Total viewshed computation on single/multi-CPU using task-parallel approach" << std::endl;
    std::cout << "  [1] -> Total viewshed computation on single/multi-CPU using sector-parallel approach" << std::endl;
    std::cout << "  [2] -> Singular/Multiple viewshed computation on single/multi-CPU (reference with DEM2k and targetPOV.csv)" << std::endl << std::endl;

#ifndef CPUONLY
    std::cout << "  [3] -> Single sector viewshed computation on single-GPU" << std::endl;
    std::cout << "  [4] -> Total viewshed computation on single/multi-GPU" << std::endl << std::endl;
#endif

    std::cout << "Available execution modes for VPP:" << std::endl << std::endl;;

    std::cout << " Point Set Cumulative Viewshed (PS-CVS) computation considering FoV = 360 degrees and:" << std::endl;
    std::cout << "  [5] -> 5 (L*) POVs" << std::endl;
    std::cout << "  [6] -> 5 (L) and 1 (V*) POVs" << std::endl;
    std::cout << "  [7] -> 5 (L) and 2 (V) POVs" << std::endl;
    std::cout << "  [8] -> 5 (L), 2 (V) and 12 (IS*) POVs" << std::endl << std::endl;
    std::cout << " Point Line Cumulative Viewshed (PL-CVS) computation considering FoV = 84 degrees and:" << std::endl;
    std::cout << "  [9] -> 235 POVs including all (L), (V), (IS) and (IN*) POVs" << std::endl << std::endl;

    std::cout << "*(L): logistic" << std::endl;
    std::cout << "*(V): getBlurParameter viewshed" << std::endl;
    std::cout << "*(IS): centroid of the isolated areas" << std::endl;
    std::cout << "*(IN): intermediate" << std::endl;

    std::cout << "  [10] -> 525 POVs including all (L), (V), (IS) and (IN*) POVs" << std::endl << std::endl;

    if(runmode==-1) {
        std::cout << std::endl << "Please enter the target execution mode (default->1): ";
        if (std::cin.peek() != '\n')
            std::cin >> runmode;
        //std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

}


void getResourcesInfo()
{
    gpu = new GpuInterface(dimy, dimx);
    nGPUs = 0;
    gpu->GetNumberGPUs(nGPUs);
#ifdef OPENMP
    nCPUs=omp_get_num_procs();
#else
    nCPUs=1;
#endif
}


void setresources()
{
#ifdef OPENMP
    if(nthreads==-1) {
        {
            nthreads = omp_get_num_procs();
            printf("Run mode 0, sector parallel cpu, with %d threads\n",nthreads);

        }
        if(runmode==1)
        {
            nthreads = omp_get_num_procs();
            while((180%nthreads))nthreads--;
            printf("Run mode 1, sector parallel cpu, with %d threads\n",nthreads);

        }
        if(runmode==4)nthreads=nGPUs;
    }

    if(runmode==0)while((180%nthreads))nthreads--;
    printf("Now, nthreads set to %d\n",nthreads);
    omp_set_num_threads(nthreads);
#else
    nthreads=1;
#endif

}

void configure_v2(int argc, char *argv[]) {
    check_args_v2(argc,argv);
    inputD=h_DEM = new float[dim];
    readBilRaster(filename, h_DEM );
    mododibujo = mododibujo & !modotiming;          // Solo dibujo s
    surScale=M_PI/(360*step*step);
    POVh=obsheight/step;


    getResourcesInfo();
    if(nthreads==-1)nthreads=nCPUs;
    setresources();

    if(runmode!=0)allocate(); //No en la nueva version


    if(verbose) {
        printf("%d CPUs and %d GPUs found. nthreads set to %d\n", nCPUs, nGPUs, nthreads);
        printf("Allocating DEM (from %s, with filesize: %dx%d)\n", filename, dimx, dimy);
    }
    }



void check_args_v2(int argc, char **argv){

    char cadena[100];
    bool ayuda = false;
    towernotgiven = true;
    if (IS_BIG_ENDIAN) {
        printf("Big Endian architectures not supported now\n");
        exit(0); // Comment this line if you can test with your own big endian models
    }
    endianOK = true;

    if (argc == 1) ayuda = true;
        // Aqui se lee el fichero
    else ayuda = !check_file_v2(argv[1]);

    if (argc == 2) return;
    int j;


    for (int i = 2; i < argc; i++) {
        if (argv[i][0] != '-') {
            ayuda=true;
            break;
        }
        switch (argv[i][1])
        {
            case 'H':
                obsheight = strtod(argv[i]+2,NULL);
                if (verbose) printf("Height: %f\n", (float)obsheight);
                break;
            case 'v':
                verbose = true;
                break;
            case 'q':
                silent = true;
                break;
            case 'F':
                fullstore = true;
                break;
            case 'm':
                inputmask = true;
                j = sscanf(argv[i], "-m%s", maskname);
                if (strlen(maskname) == 0) sprintf(maskname, "%s/mask.dat", INPUTDIR);
                break;
            case 't':
                j = sscanf(argv[i], "-t%d", &mode);
                //printf("Modo: %d\n", mode);
                if (j == 0) mode = 0;
                break;
            case 'r':
                j = sscanf(argv[i], "-r%d", &runmode);
                //printf("Modo: %d\n", mode);
                if (j == 0) runmode = -1;
                break;
            case 'g':
                j = sscanf(argv[i], "-g%d", &deviceIndex);
                break;
            case 's':
                j = sscanf(argv[i], "-s%d", &maxSector);
                break;
            case 'p':
                sscanf(argv[i], "-p%d", &nthreads);
                break;
            case '?':
                break;
            default:
                ayuda = true;
                break;
        }
    }

    if (ayuda) {
        printf("Usage: %s model [-v] [-q] [-ttype][-Hheight][-mmaskfilename] [-F] [-nitem]\n\n"
               "-v Verbose mode\n"
               "-q Silent mode\n"
               "-tN Usage mode:\n"
               "\t N=0 Total viewshed (default)\n"
               "\t N=2 Single tower viewshed (no included yet)\n"
               "\t N=3 Track Viewshed (no included yet)\n"
               "\t N=6 Find sequential towers (cell coverage algorithm)  (no included yet)\n"
               "\t N=7 Isolation index  (no included yet)\n"
               "\t N=8 Horizons  (no included yet)\n"
               "-rN Running mode:\n"
               "\t N=0 CPU v1\n"
               "\t N=0 CPU v2 (default)\n"
               "\t N=0 GPU v1\n"
               "\t N=0 GPU v2\n"

               "-sN Max sector (179?)\n"
               "-Hheight Observer's height in meters (default, 1.5m)\n"
               "-m Use default mask file (input/mask.dat for area of interest\n"
               "-mfile Use file for area of interest\n"

               "-nitem In single-tower and several-tower (tracks) viewshed execution modes, specifies the track:\n"
               "\t item:utmn,utme,name\n"
               "\t item:index,name\n"
               "\t item:index\n"
               "\t Use custom function to assign indexes to towers\n\n"
               "",argv[0]); exit(0);
    }
    //exit(0);
}


//-----------------------------------------------------------------------------------------------------
//                    Memory functions
//-----------------------------------------------------------------------------------------------------

void allocThreadSectorData(float **&skwDEM, float **&rotatedVS) {
    for (int y = 0; y < dimy; y++) {
        skwDEM[y] = new float[dimx];
        skwDEM[y + dimy] = new float[dimx];
    }
    // Instantiation
    for (int y = 0; y < dimy; y++) {
        rotatedVS[y] = new float[dimx];
        rotatedVS[y + dimy] = new float[dimx];
    }

}

void deallocThreadSectorData(float **skwDEM, float **rotatedVS, int *destiny, float * ratio, float *sectorVS)
{
for (int i = 0; i < 2 * dimy; i++)
    delete[] rotatedVS[i];
delete[] rotatedVS;
delete destiny;
delete ratio;
delete sectorVS;
}

void allocSharedData(float **totalVS)
{
}

void deallocSharedData(float **totalVS)
{
    for (int y = 0; y < dimy; y++)
        delete totalVS[y];
    delete[] totalVS;
}

void resetSharedData(float **totalVS, float *floatVS)
{
    // Initialization
    for (int y = 0; y < dimy; y++)
        for (int x = 0; x < dimx; x++) {
            // sectorVS[y * dimy + x] = 0.0;
            totalVS[y][x] = 0.0;
            floatVS[y * dimx + x] = 0.0;
        }

}

void resetThreadSectorData(float **skwDEM, float **rotatedVS) {
    for (int y = 0; y < dimy; y++)
        for (int x = 0; x < dimx; x++)
            rotatedVS[y][x] = 0.0;
    for (int y = 0; y < dimy; y++)
        for (int x = 0; x < dimx; x++)
            rotatedVS[y + dimy][x] = 0.0;
    for (int y = 0; y < dimy; y++)
        for (int x = 0; x < dimx; x++)
            skwDEM[y][x] = 0.0;
    for (int y = 0; y < dimy; y++)
        for (int x = 0; x < dimx; x++)
            skwDEM[y + dimy][x] = 0.0;

}

void allocate() {

    double min=FLT_MAX,max=FLT_MIN;
    float *hdem = nullptr;
    heights=new double[dim];
    for(int i=0;i<dim;i++)heights[i]=h_DEM[i];
    //heights = readHeights(filename,false,true);
    //float *h_DEM=(float *)malloc(dim*sizeof (float));
    //for(int i=0;i<dim;i++)heights[i]=h_DEM[i];

    //printf(" %d %d %d %d   \n",(int)heights[0],(int)heights[dimx-1],(int)heights[dim-dimx],(int)heights[dim-1]);

    profileX = (double **)malloc(dimx * sizeof(double *));
    for (int i = 0; i < dimy; i++)
        profileX[i] = (double *)malloc(dimx * sizeof(double));
    //for(int i=1;i<10000;i++)inversos[i]=1/(float)i; //experimento fallido
//To Do #NoCuadrado
#ifdef NOCUADRADO
    profileY = (double **)malloc(dimy * sizeof(double *));
    for (int i = 0; i < dimx; i++)
        profileY[i] = (double *)malloc(sqrt(dim) * sizeof(double));
#endif

    resultado = (float **)malloc((dimy + dimx) * sizeof(float *));
    for (int i = 0; i < dimy; i++) {
        resultado[i] = (float *)malloc(dimx * sizeof(float));
        resultado[dimy + i] = (float *)malloc(dimx * sizeof(float));
        for (int j = 0; j < dimx; j++) {
            resultado[dimy + i][j] = heights[dimx * i + j];
            resultado[i][j] = 0;
            min= heights[dimx * i + j]<min? heights[dimx * i + j]:min;
            max= heights[dimx * i + j]>max? heights[dimx * i + j]:max;
        }
    }
    printf("Extreme values for height (/step): %f - %f\n",min,max);


    // dimx????
    desti = new int[dimx];
    rat = new float[dimx];


}

void deallocate() {
    for (int i = 0; i < dimx; i++) free(profileX[i]);
    free(profileX);
#ifdef NOCUADRADO
    for (int i = 0; i < dimx; i++) free(profileY[i]);
    free(profileY);
#endif
    free(heights);
    for (int i = 0; i < dimy + dimx; i++)
        free (resultado[i]);
    free (resultado);

    delete[] desti;
    delete[] rat;
    //delete skewer;
    delete gpu;
}

void deallocateV3(int skewAlgorithm) {
    if(skewAlgorithm==0)cpu->FreeHostMemory(outD,inData.input0,inData.input1,inData.input2,inData.input3);
    if(skewAlgorithm==1)cpu->FreeHostMemory(outD,inData.input0,inData.input1,inData.input2,inData.input3);
    if(skewAlgorithm==0)free(inputD);
    if(skewAlgorithm==1)free(inputD);
    delete cpu;
    delete gpu;
}

void allocateV3(int skewAlgorithm)
{

    // AllocDEMHost (in auxf.cu) allocate arrays in CPU in a better way than malloc, if it's going to be
    // used in CUDA  ("pinned" memory)

    //
    int dataSize=sizeof(float);
    if(skewAlgorithm==0)dataSize=sizeof(float);
    if(skewAlgorithm==1)dataSize=sizeof(float);

    cpu->AllocDEMHost(outD,inData.input0,inData.input1,inData.input2,inData.input3,dim);
    memcpy(inData.input0,inputD,dim*dataSize); //Move input data to pinned memory
    memset(outD,0,dim*dataSize);

    // Cambiar, si no es sdem ni blur
    inData=  skewEngine<float>::prepare(&inData,dimx,dimy);// Rotated and mirror
}
